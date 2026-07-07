import { BrainCircuit } from 'lucide-react';

export function Header() {
  return (
    <header className="header">
      <div className="brand">
        <div className="brandIcon"><BrainCircuit size={28} /></div>
        <div>
          <h1>Sudoku Solver</h1>
          <p>Upload a picture of your Sudoku puzzle and inspect every CV phase.</p>
        </div>
      </div>
      <div className="badge">OpenCV + PyTorch</div>
    </header>
  );
}
