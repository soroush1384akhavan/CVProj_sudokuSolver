import { mediaUrl } from '../api/sudokuApi';
import type { PhaseImage } from '../types';

interface Props {
  phases: PhaseImage[];
}

export function PhasePanel({ phases }: Props) {
  if (phases.length === 0) {
    return (
      <aside className="phasePanel empty">
        <h2>CV Phase Results</h2>
        <p>After processing an image, every phase output will appear here.</p>
      </aside>
    );
  }

  return (
    <aside className="phasePanel">
      <div className="sectionTitle">
        <h2>CV Phase Results</h2>
        <span>{phases.length} outputs</span>
      </div>
      <div className="phaseList">
        {phases.map((phase) => {
          const url = mediaUrl(phase.image_url);
          return (
            <article key={`${phase.key}-${phase.image_url}`} className="phaseCard">
              <div className="phaseText">
                <h3>{phase.title}</h3>
                <p>{phase.description}</p>
              </div>
              {url ? <a href={url} target="_blank" rel="noreferrer"><img src={url} alt={phase.title} /></a> : null}
            </article>
          );
        })}
      </div>
    </aside>
  );
}
