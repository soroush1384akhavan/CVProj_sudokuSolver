import { Eraser, Play, RefreshCcw, ScanLine } from 'lucide-react';

interface Props {
  isProcessing: boolean;
  isSolving: boolean;
  onProcess: () => void;
  onSolve: () => void;
  onReset: () => void;
  onClear: () => void;
}

export function Controls({ isProcessing, isSolving, onProcess, onSolve, onReset, onClear }: Props) {
  return (
    <div className="controls">
      <button className="primaryButton" onClick={onProcess} disabled={isProcessing || isSolving}>
        <ScanLine size={18} /> {isProcessing ? 'Processing...' : 'Process Image'}
      </button>
      <button className="successButton" onClick={onSolve} disabled={isProcessing || isSolving}>
        <Play size={18} /> {isSolving ? 'Solving...' : 'Solve Puzzle'}
      </button>
      <button className="secondaryButton" onClick={onReset} disabled={isProcessing || isSolving}>
        <RefreshCcw size={18} /> Reset
      </button>
      <button className="dangerButton" onClick={onClear} disabled={isProcessing || isSolving}>
        <Eraser size={18} /> Clear
      </button>
    </div>
  );
}
