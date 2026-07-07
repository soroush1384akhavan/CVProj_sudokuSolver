import { KeyboardEvent } from 'react';

interface Props {
  row: number;
  col: number;
  value: number;
  original: boolean;
  lowConfidence: boolean;
  confidence?: number;
  onChange: (row: number, col: number, value: number) => void;
}

export function SudokuCell({ row, col, value, original, lowConfidence, confidence, onChange }: Props) {
  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    const key = event.key;
    if (/^[1-9]$/.test(key)) {
      event.preventDefault();
      onChange(row, col, Number(key));
      focusNext(row, col);
    }
    if (key === 'Backspace' || key === 'Delete' || key === '0') {
      event.preventDefault();
      onChange(row, col, 0);
    }
    if (['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp'].includes(key)) {
      event.preventDefault();
      const next = {
        ArrowRight: [row, Math.min(8, col + 1)],
        ArrowLeft: [row, Math.max(0, col - 1)],
        ArrowDown: [Math.min(8, row + 1), col],
        ArrowUp: [Math.max(0, row - 1), col],
      }[key] as number[];
      focusCell(next[0], next[1]);
    }
  }

  function focusCell(r: number, c: number) {
    document.querySelector<HTMLInputElement>(`input[data-cell="${r}-${c}"]`)?.focus();
  }

  function focusNext(r: number, c: number) {
    const index = r * 9 + c;
    const next = Math.min(80, index + 1);
    focusCell(Math.floor(next / 9), next % 9);
  }

  const className = [
    'sudokuCell',
    original ? 'original' : 'editable',
    lowConfidence ? 'lowConfidence' : '',
    col === 2 || col === 5 ? 'thickRight' : '',
    row === 2 || row === 5 ? 'thickBottom' : '',
  ].filter(Boolean).join(' ');

  return (
    <input
      data-cell={`${row}-${col}`}
      className={className}
      value={value === 0 ? '' : value}
      inputMode="numeric"
      maxLength={1}
      title={confidence !== undefined ? `confidence: ${confidence}` : undefined}
      onKeyDown={handleKeyDown}
      onChange={(event) => {
        const raw = event.target.value;
        if (raw === '') onChange(row, col, 0);
        if (/^[1-9]$/.test(raw)) onChange(row, col, Number(raw));
      }}
    />
  );
}
