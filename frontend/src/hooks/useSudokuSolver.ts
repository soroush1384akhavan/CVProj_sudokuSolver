import { useMemo, useState } from 'react';
import { predictSudoku, solveSudoku } from '../api/sudokuApi';
import type { DigitLanguage } from '../api/sudokuApi';
import type { Board, PhaseImage } from '../types';
import { cloneBoard, emptyBoard } from '../utils/board';

export function useSudokuSolver() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [grid, setGrid] = useState<Board>(emptyBoard());
  const [originalGrid, setOriginalGrid] = useState<Board>(emptyBoard());
  const [confidence, setConfidence] = useState<number[][]>(emptyBoard());
  const [lowConfidenceCells, setLowConfidenceCells] = useState<Array<{ row: number; col: number }>>([]);
  const [phases, setPhases] = useState<PhaseImage[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSolving, setIsSolving] = useState(false);
  const [message, setMessage] = useState('Upload a Sudoku image to begin.');
  const [modelStatus, setModelStatus] = useState('Model status will appear here.');
  const [language, setLanguage] = useState<DigitLanguage>('en');

  const lowConfidenceSet = useMemo(() => {
    return new Set(lowConfidenceCells.map((cell) => `${cell.row}-${cell.col}`));
  }, [lowConfidenceCells]);

  function chooseImage(file: File) {
    setSelectedImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    setMessage('Image selected. Click Process Image.');
  }

  async function processImage() {
    if (!selectedImage) {
      setMessage('Please select an image first.');
      return;
    }

    setIsProcessing(true);
    setMessage('Processing image...');

    try {
      const result = await predictSudoku(selectedImage, language);

      setRunId(result.run_id);
      setGrid(result.board);
      setOriginalGrid(cloneBoard(result.board));
      setConfidence(result.confidence);
      setLowConfidenceCells(result.low_confidence_cells);
      setPhases(result.phases);
      setMessage(result.message);
      setModelStatus(result.model_status);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Failed to process image.');
    } finally {
      setIsProcessing(false);
    }
  }

  async function solve() {
    setIsSolving(true);
    setMessage('Solving puzzle...');

    try {
      const result = await solveSudoku(grid, runId, originalGrid);

      if (result.success && result.solved_board) {
        setGrid(result.solved_board);

        if (result.phases.length > 0) {
          setPhases((prev) => [
            ...prev.filter((phase) => !phase.key.startsWith('solved')),
            ...result.phases,
          ]);
        }
      }

      setMessage(result.message);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Failed to solve puzzle.');
    } finally {
      setIsSolving(false);
    }
  }

  function updateCell(row: number, col: number, value: number) {
    setGrid((prev) => {
      const next = cloneBoard(prev);
      next[row][col] = value;
      return next;
    });
  }

  function resetGrid() {
    setGrid(cloneBoard(originalGrid));
    setMessage('Grid reset to detected board.');
  }

  function clearAll() {
    setSelectedImage(null);
    setPreviewUrl(null);
    setRunId(null);
    setGrid(emptyBoard());
    setOriginalGrid(emptyBoard());
    setConfidence(emptyBoard());
    setLowConfidenceCells([]);
    setPhases([]);
    setMessage('Cleared. Upload a new image to begin.');
    setModelStatus('Model status will appear here.');
  }

  return {
    selectedImage,
    previewUrl,
    runId,
    grid,
    originalGrid,
    confidence,
    lowConfidenceSet,
    phases,
    isProcessing,
    isSolving,
    message,
    modelStatus,
    language,
    setLanguage,
    chooseImage,
    processImage,
    solve,
    updateCell,
    resetGrid,
    clearAll,
  };
}