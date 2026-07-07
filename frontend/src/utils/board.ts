import type { Board } from '../types';

export function emptyBoard(): Board {
  return Array.from({ length: 9 }, () => Array.from({ length: 9 }, () => 0));
}

export function cloneBoard(board: Board): Board {
  return board.map((row) => [...row]);
}

export function boardToText(board: Board): string {
  return board.map((row) => row.join(' ')).join('\n');
}

export function hasValue(board: Board): boolean {
  return board.some((row) => row.some((value) => value !== 0));
}
