# Proyecto
En el archivo notebooks/exercises/Titanic.ipynb se encuentra el ejercicio que tiene que resolver. Al finalizar el ejercicio, Github Classroom le asignará su calificación.

# Funcionamiento del repositorio
En la carpeta `notebooks` se encuentran los ejercicios que se requieren resolver. Cada módulo contiene una serie de preguntas que necesita resolver dependiendo del contexto del problema. En general, los tipos de respuesta que se le puede solicitar son los siguientes:

- **Ejercicio de reflexión**: Se le pedirá que reflexione acerca de la pregunta propuesta. Al finalizar, podrá ejecutar la celda de manera exitosa. Note que no se necesita resolver el ejercicio, solamente hace falta que ejecute la celda.

- **Revisión de una variable**: Se le pedirá que se le asigne un valor a una variable con un nombre específico. Es importante no modificar el nombre de la variable para que la librería pueda calificar la pregunta.

- **Revisión de una función**: Se le pedirá que desarrolle una función con un nombre específico. Es importante no modificar el nombre de la función para que la librería pueda calificar la pregunta.

Para que la librería califique los módulos, solamente necesita ejecutar el Jupyter Notebook. Al terminar se le indicará cuales de sus respuestas son correctas y cuales no son correctas.

Al inicio de cada módulo habrá un bloque de código para importar las librerías necesarias para calificar el ejercicio. Para poder proporcionar la calificación al ejercicio, es necesario que este bloque de código no sea modificado.

Al final de cada celda existe una linea de código con una variable, generalmente con el nombre `q*` donde `*` representa el número de la pregunta. Es necesario mantener esta línea de código para poder proporcionarle una calificación al final del ejercicio.

# Revisar los ejercicios

Para revisar todos los ejercicios se necesita ejecutar el siguiente comando:

```
make
```

Para revisar un ejercicio específico se necesita ejecutar el siguiente comando:

```
make conv-test JFILE=Nombre_Ejercicio PFILE=Nombre_Prueba
```

Para revisar un problema específico de un ejercicio específico se necesita ejecutar el siguiente comando:

```
make conv-test-func JFILE=Nombre_Ejercicio PFILE=Nombre_Prueba FUNCTION="Nombre_Problema"
```

Para excluir un problema específico dentro de la prueba de un ejercicio específico se necesita ejecutar el siguiente comando:

```
make conv-test-func JFILE=Nombre_Ejercicio PFILE=Nombre_Prueba FUNCTION="not Nombre_Problema"
```