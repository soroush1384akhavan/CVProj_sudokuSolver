import { Header } from './components/Header';
import { ImageUploader } from './components/ImageUploader';
import { SudokuGrid } from './components/SudokuGrid';
import { PhasePanel } from './components/PhasePanel';
import { Controls } from './components/Controls';
import { useSudokuSolver } from './hooks/useSudokuSolver';
import './styles.css';

export default function App() {
  const solver = useSudokuSolver();

  return (
    <main className="appShell">
      <Header />
      <div className="layout">
        <section className="leftColumn">
          <ImageUploader previewUrl={solver.previewUrl} onImageSelected={solver.chooseImage} />
          <Controls
            isProcessing={solver.isProcessing}
            isSolving={solver.isSolving}
            onProcess={solver.processImage}
            onSolve={solver.solve}
            onReset={solver.resetGrid}
            onClear={solver.clearAll}
          />
          <div className="messageBox">
            <strong>Status:</strong> {solver.message}
            <br />
            <strong>Model:</strong> {solver.modelStatus}
          </div>
        </section>

        <section className="centerColumn">
          <div className="panelHeader">
            <h2>Detected Sudoku Grid</h2>
            <p>Correct any wrong cells before solving.</p>
          </div>
          <SudokuGrid
            grid={solver.grid}
            originalGrid={solver.originalGrid}
            confidence={solver.confidence}
            lowConfidenceSet={solver.lowConfidenceSet}
            onCellChange={solver.updateCell}
          />
          {solver.lowConfidenceSet.size > 0 ? (
            <div className="warningBox">Highlighted cells have low confidence. Please verify and correct them if needed.</div>
          ) : null}
        </section>

        <PhasePanel phases={solver.phases} />
      </div>
    </main>
  );
}
