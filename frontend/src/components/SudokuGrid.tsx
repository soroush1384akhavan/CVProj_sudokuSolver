import type { Board } from '../types';
import { SudokuCell } from './SudokuCell';

interface Props {
  grid: Board;
  originalGrid: Board;
  confidence: number[][];
  lowConfidenceSet: Set<string>;
  onCellChange: (row: number, col: number, value: number) => void;
}

export function SudokuGrid({ grid, originalGrid, confidence, lowConfidenceSet, onCellChange }: Props) {
  return (
    <div className="gridWrap">
      <div className="sudokuGrid">
        {grid.map((row, r) => row.map((value, c) => (
          <SudokuCell
            key={`${r}-${c}`}
            row={r}
            col={c}
            value={value}
            original={originalGrid[r]?.[c] !== 0}
            lowConfidence={lowConfidenceSet.has(`${r}-${c}`)}
            confidence={confidence[r]?.[c]}
            onChange={onCellChange}
          />
        )))}
      </div>
      <div className="legend">
        <span><i className="dot originalDot" /> detected/given</span>
        <span><i className="dot editableDot" /> editable/solved</span>
        <span><i className="dot lowDot" /> low confidence</span>
      </div>
    </div>
  );
}
