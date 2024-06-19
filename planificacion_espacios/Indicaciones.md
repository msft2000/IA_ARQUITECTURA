# Instrucciones para Utilizar la Herramienta en el Intérprete de Python de Dynamo

Para utilizar la herramienta, es necesario instalar un paquete adicional en el intérprete de Python de Dynamo. Sigue estos pasos:

1. **Buscar Carpeta**  
   Busca una carpeta cuyo nombre contenga `python embedded amd64` en la ruta `C:/Users/Username/Appdata/Local`.

2. **Modificar Archivo**  
   Abre el archivo `python39._pth` y quita el `#` de la línea `import site`.

3. **Copiar Archivo get-pip**  
   Copia el archivo `get-pip.py` a esa carpeta.

4. **Abrir CMD**  
   Abre el símbolo del sistema (cmd) y ubica la carpeta.

5. **Ejecutar Comando**  
   Corre el comando:
   ```bash
   python get-pip.py

6. **Instalar Paquetes**
	Corre el comando:
	`.\Scripts\pip install paquete`
7. **Instalar el Paquete Shapely**
	Corre el comando:
	`.\Scripts\pip install shapely==1.8.0`

**Nota:**
	Dentro de Dynamo se usa estos comandos para indicar correctamente el interprete
	`localapp = os.getenv(r'LOCALAPPDATA')`
	`sys.path.append(os.path.join(localapp, r'python-3.8.3-embed-amd64\Lib\site-packages'))`