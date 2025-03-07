import os
import torch
import csv

def frames_to_timestamp(frame_num, fps=25):
    """Convierte número de frame a timestamp en formato HH:MM:SS:FF"""
    total_seconds = frame_num / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    frames = int((total_seconds * fps) % fps)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

def get_confidence_score(score_tensor):
    """Convierte el tensor de predicción multidimensional en un valor de confianza"""
    if isinstance(score_tensor, torch.Tensor):
        # Promediamos todos los valores del tensor para obtener un score
        return score_tensor.mean().item()
    return score_tensor

def save_speaker_annotations(args, tracks, pred, min_prob=0.5):
    """
    Guarda las anotaciones de los hablantes en formato TXT y CSV
    args: argumentos del programa
    tracks: información de tracking de caras
    pred: predicciones del modelo
    """
    # Crear directorio para los archivos
    txt_dir = os.path.join(args.savePath, 'asd_txt')
    os.makedirs(txt_dir, exist_ok=True)
    
    # Archivos de salida
    output_txt = os.path.join(txt_dir, 'speaker_annotations.txt')
    output_csv = os.path.join(txt_dir, 'speaker_annotations.csv')
    
    # Procesar tracks y predicciones
    annotations = []
    for tidx, track in enumerate(tracks):
        score = pred[tidx]
        
        # Obtener frames donde el speaker está hablando
        speaking_segments = []
        start_frame = None
        prev_speaking = False
        
        for fidx, frame in enumerate(track['frame']):
            current_score = get_confidence_score(score[fidx])
            is_speaking = current_score >= min_prob  # Umbral de confianza ajustable
            
            if is_speaking and not prev_speaking:
                start_frame = frame
            elif not is_speaking and prev_speaking:
                end_frame = frame - 1
                conf_score = get_confidence_score(score[fidx-1])
                speaking_segments.append((start_frame, end_frame, conf_score))
            
            prev_speaking = is_speaking
        
        # Manejar el último segmento si termina hablando
        if prev_speaking:
            end_frame = track['frame'][-1]
            conf_score = get_confidence_score(score[-1])
            speaking_segments.append((start_frame, end_frame, conf_score))
            
        # Guardar segmentos
        for start, end, conf in speaking_segments:
            annotations.append({
                'speaker': tidx + 1,
                'confidence': conf,
                'start_frame': start,
                'end_frame': end
            })
    
    # Ordenar por tiempo de inicio
    annotations.sort(key=lambda x: x['start_frame'])
    
    # Escribir archivo TXT
    with open(output_txt, 'w') as f:
        for ann in annotations:
            start_time = frames_to_timestamp(ann['start_frame'])
            end_time = frames_to_timestamp(ann['end_frame'])
            
            f.write(f"Speaker {ann['speaker']} - {ann['confidence']:.2f}\n")
            f.write(f"{start_time} --> {end_time}\n\n")
    
    # Escribir archivo CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Escribir encabezados
        writer.writerow(['Speaker', 'Start_Frame', 'End_Frame', 'Start_Time', 'End_Time', 'Confidence'])
        
        # Escribir datos
        for ann in annotations:
            start_time = frames_to_timestamp(ann['start_frame'])
            end_time = frames_to_timestamp(ann['end_frame'])
            
            writer.writerow([
                f"Speaker {ann['speaker']}", 
                ann['start_frame'],
                ann['end_frame'],
                start_time,
                end_time,
                f"{ann['confidence']:.2f}"
            ])