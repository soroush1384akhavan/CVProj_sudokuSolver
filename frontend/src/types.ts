export type Board = number[][];

export interface PhaseImage {
  key: string;
  title: string;
  description: string;
  image_url: string | null;
}

export interface PredictResponse {
  success: boolean;
  run_id: string;
  board: Board;
  confidence: number[][];
  low_confidence_cells: Array<{ row: number; col: number }>;
  phases: PhaseImage[];
  message: string;
  model_status: string;
}

export interface SolveResponse {
  success: boolean;
  solved_board: Board | null;
  phases: PhaseImage[];
  message: string;
}
