import type { Board, PredictResponse, SolveResponse } from '../types';

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export async function predictSudoku(image: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append('image', image);

  const response = await fetch(`${API_BASE_URL}/api/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail || 'Failed to process the image.');
  }

  return response.json();
}

export async function solveSudoku(board: Board, runId: string | null, originalBoard: Board | null): Promise<SolveResponse> {
  const response = await fetch(`${API_BASE_URL}/api/solve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ board, run_id: runId, original_board: originalBoard }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail || 'Failed to solve the puzzle.');
  }

  return response.json();
}

export function mediaUrl(path: string | null): string | null {
  if (!path) return null;
  if (path.startsWith('http')) return path;
  return `${API_BASE_URL}${path}`;
}
