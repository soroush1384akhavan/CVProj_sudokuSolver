import { ChangeEvent, DragEvent, useRef, useState } from 'react';
import { ImagePlus, UploadCloud } from 'lucide-react';

interface Props {
  previewUrl: string | null;
  onImageSelected: (file: File) => void;
}

export function ImageUploader({ previewUrl, onImageSelected }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);

  function handleFile(file: File | undefined) {
    if (!file) return;
    if (!file.type.startsWith('image/')) return;
    onImageSelected(file);
  }

  function onInputChange(event: ChangeEvent<HTMLInputElement>) {
    handleFile(event.target.files?.[0]);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    handleFile(event.dataTransfer.files?.[0]);
  }

  return (
    <section
      className={`uploader ${dragging ? 'dragging' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input ref={inputRef} type="file" accept="image/*" onChange={onInputChange} hidden />
      {previewUrl ? (
        <div className="previewWrap">
          <img src={previewUrl} alt="Selected Sudoku" />
          <button type="button" className="ghostButton"><ImagePlus size={18} /> Change Image</button>
        </div>
      ) : (
        <div className="uploadHint">
          <UploadCloud size={44} />
          <h2>Drag and drop your Sudoku image</h2>
          <p>Or click to browse</p>
        </div>
      )}
    </section>
  );
}
