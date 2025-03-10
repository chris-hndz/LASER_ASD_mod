import psutil
import GPUtil
import time
import os
import csv
import datetime
from threading import Thread, Event

# Función para formatear números con coma decimal
def format_number(number, decimals=2):
    """Formatea un número usando coma como separador decimal"""
    return str(round(number, decimals)).replace('.', ',')

class ResourceMonitor:
    def __init__(self, output_folder, video_name, interval=1.0):
        """
        Inicializa el monitor de recursos
        
        Args:
            output_folder: Carpeta donde se guardará el archivo CSV
            video_name: Nombre del video que se está procesando
            interval: Intervalo en segundos entre mediciones
        """
        self.output_folder = output_folder
        self.video_name = video_name
        self.interval = interval
        self.stop_event = Event()
        self.monitor_thread = None
        self.start_time = None
        
        # Crear la carpeta de salida si no existe
        os.makedirs(os.path.join(output_folder, 'asd_txt'), exist_ok=True)
        
        # Definir la ruta del archivo CSV
        self.csv_path = os.path.join(output_folder, 'asd_txt', f'resources_{video_name}.csv')
        
    def _monitor_resources(self):
        """Función que ejecuta el monitoreo en segundo plano"""
        # Crear archivo CSV con punto y coma como separador y escribir encabezados
        with open(self.csv_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(['timestamp', 'elapsed_time_s', 'cpu_percent', 'memory_used_mb', 'gpu_utilization', 'gpu_memory_mb'])
            
            while not self.stop_event.is_set():
                # Obtener métricas de CPU y RAM
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                memory_used_mb = memory_info.used / (1024 * 1024)  # Convertir bytes a MB
                
                # Obtener métricas de GPU si está disponible
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Usar la primera GPU
                        gpu_util = gpu.load * 100  # Convertir a porcentaje
                        gpu_memory_mb = gpu.memoryUsed  # Ya viene en MB
                    else:
                        gpu_util = 0
                        gpu_memory_mb = 0
                except Exception:
                    gpu_util = 0
                    gpu_memory_mb = 0
                
                # Calcular tiempo transcurrido
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                # Registrar timestamp actual
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Formatear números con coma decimal
                elapsed_time_str = format_number(elapsed_time)
                cpu_percent_str = format_number(cpu_percent)
                memory_used_mb_str = format_number(memory_used_mb)
                gpu_util_str = format_number(gpu_util)
                gpu_memory_mb_str = format_number(gpu_memory_mb)
                
                # Escribir datos en CSV
                writer.writerow([timestamp, elapsed_time_str, cpu_percent_str, memory_used_mb_str, gpu_util_str, gpu_memory_mb_str])
                
                # Esperar hasta el próximo intervalo
                csv_file.flush()  # Asegurar que los datos se escriban inmediatamente
                time.sleep(self.interval)
    
    def start(self):
        """Inicia el monitoreo de recursos"""
        self.start_time = time.time()
        self.stop_event.clear()
        self.monitor_thread = Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"Monitoreo de recursos iniciado. Se guardará en: {self.csv_path}")
        
    def stop(self):
        """Detiene el monitoreo de recursos"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_event.set()
            self.monitor_thread.join(timeout=5)
            print(f"Monitoreo de recursos finalizado. Datos guardados en: {self.csv_path}")
            return self.csv_path
        return None