# Magnificador de Pulso Médico

## Descripción General
El Magnificador de Pulso Médico es un proyecto en Python diseñado para procesar fotogramas de video en tiempo real para magnificar las señales de pulso desde la frente del usuario. Utilizando técnicas avanzadas de procesamiento de imágenes, esta aplicación captura la entrada de video, detecta características faciales y aplica magnificación a la región de interés (ROI) para visualizar cambios en el pulso.

## Características
- Procesamiento de video en tiempo real usando OpenCV
- Visualización de flujo óptico para seguimiento de movimiento
- Magnificación de imagen basada en pirámide laplaciana
- Representación gráfica de la intensidad promedio y ritmo cardíaco estimado (BPM)
- Auto-ajuste inteligente de parámetros alpha y lambda
- Detección automática de ROI en la frente
- Procesamiento optimizado con concurrent futures

## Instalación
Para configurar el proyecto, asegúrate de tener Python instalado en tu máquina. Luego, sigue estos pasos:

1. Clona el repositorio:
   ```
   git clone https://github.com/nwtn777/medical_motion_magnification.git
   cd medical_motion_magnification
   ```

2. Instala los paquetes requeridos:
   ```
   pip install -r requirements.txt
   ```

## Uso
Para ejecutar la aplicación, usa el siguiente comando en tu terminal:
```
python src/medical_optimized_concurrent_futures.py
```

Argumentos opcionales:
- `--fps`: Frames por segundo (default: 20)
- `--alpha`: Sensibilidad de magnificación (default: 200)
- `--lambda_c`: Lambda de corte (default: 20)
- `--fl`: Frecuencia baja (default: 0.5)
- `--fh`: Frecuencia alta (default: 3.0)
- `--auto_tune`: Habilitar auto-ajuste de parámetros alpha y lambda

Ejemplo con argumentos personalizados:
```
python src/medical_optimized_concurrent_futures.py --fps 30 --alpha 150 --auto_tune
```

Asegúrate de que tu cámara esté conectada y accesible. La aplicación abrirá una ventana mostrando la transmisión de video con la ROI del pulso magnificada.

## Requisitos
El proyecto requiere los siguientes paquetes de Python:
- opencv-python
- scipy
- scikit-image
- numpy
- matplotlib
- pyrtools
- PyQt5

## Estado del Proyecto
El proyecto está en desarrollo activo. Características actuales y limitaciones:
- Auto-ajuste de parámetros para mejorar la detección del pulso
- ⚠️ AVISO IMPORTANTE: La medición del BPM es altamente inconsistente y no debe utilizarse para propósitos médicos
  - Las lecturas pueden variar significativamente entre mediciones
  - Los valores de BPM pueden no corresponder a la realidad
  - La función está en fase experimental y requiere mayor desarrollo
- Optimización continua del rendimiento y precisión
- Se recomienda usar la aplicación solo para propósitos de investigación y desarrollo

## Contribuciones
¡Las contribuciones son bienvenidas! No dudes en enviar un pull request o abrir un issue para mejoras o corrección de errores.

## Licencia
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.