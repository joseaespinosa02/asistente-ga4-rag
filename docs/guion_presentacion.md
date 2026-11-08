# Guion de la presentación (5-10 min)

Diapositivas: [`presentacion.pdf`](presentacion.pdf). Tiempo orientativo entre paréntesis.

**1. Portada (15 s)**
«Os presento Analytica, un asistente experto en Google Analytics 4 construido con Gemini, RAG y un agente en LangGraph con memoria.»

**2. El problema (45 s)**
- La documentación de GA4 está repartida en decenas de artículos y cambia (de «conversiones» a «eventos clave»).
- Un chatbot genérico puede inventar eventos o rutas de menú.
- Objetivo: responder solo con la documentación oficial, citar la fuente y mantener el contexto.

**3. Stack (30 s)**
Recorrer la tabla: Gemini para el LLM y los embeddings, ChromaDB como base vectorial, LangGraph para el agente, Colab para el notebook y Streamlit como bonus.

**4. Base de conocimiento (1 min)**
- 16 artículos oficiales en español → 225 fragmentos.
- Explicar el chunking: 1.000 caracteres para que cada fragmento contenga una idea completa, 150 de solapamiento para no perder información en los cortes.
- La limpieza HTML quita el ruido (menús, «¿Te ha resultado útil?»), que empeoraría los embeddings.

**5. Verificación (30 s)**
«Antes de conectar el agente comprobé que la recuperación funciona: en las 4 consultas, el primer resultado es el documento correcto.»

**6. Arquitectura (1 min)**
- Tres nodos fijos: contextualize → retrieve → generate.
- El flujo fijo garantiza que se consulta la base de conocimiento en cada pregunta.
- MMR: fragmentos relevantes y no repetidos.

**7. Reformulación (45 s)**
Idea clave: «la memoria no solo sirve para redactar, también para buscar». Sin reformular, «¿y cómo la activo?» no encuentra nada útil en la base vectorial.

**8. Memoria (30 s)**
Checkpointer de LangGraph por `thread_id`. Un hilo nuevo no recuerda nada: lo demuestro en la demo.

**9. System prompt (45 s)**
Destacar «responder solo con el contexto» como defensa contra alucinaciones y «decir cuándo no sabe».

**10. Retos técnicos (45 s)**
Contar lo que pasó de verdad: se agotó la cuota diaria de embeddings, cambiaron los nombres de los modelos y hubo errores 500. Explicar cómo se resolvió cada uno: reutilizar el índice, cambiar de modelo automáticamente y reintentar.

**11. Resultados reales: memoria (45 s)**
Enseñar la pregunta 6: la búsqueda reformulada convierte «lo que vimos al principio» en una consulta concreta, y el resumen recoge las preguntas 1 a 3 con sus fuentes.

**12. Resultados reales: hilos y límites (45 s)**
- El hilo «demo» guarda 14 mensajes; en un hilo nuevo el agente no sabe cuál fue la primera pregunta. La memoria está aislada por `thread_id`.
- La paella se rechaza y se redirige a GA4.
- Reconocer la limitación: aun así consulta ChromaDB. Mejora propuesta: un nodo *router*.

**13. Demo en vivo (2-3 min)**
Ejecutar las preguntas en Colab (o en Streamlit) en este orden:
1. ¿Qué tipos de eventos existen en GA4 y en qué se diferencian?
2. ¿Qué eventos recoge la medición mejorada?
3. **¿Y cómo la activo o desactivo?** → señalar la línea «búsqueda reformulada»
4. ¿Qué parámetros necesita el evento purchase para medir ventas?
5. ¿Puedo editar el grupo de canales predeterminado?
6. **Resúmeme lo que vimos sobre eventos al principio** → memoria de varios turnos
7. ¿Me recomiendas una receta de paella? → límites del dominio

> Consejo: ejecuta el notebook completo antes de la clase. Así el índice ya está construido y la demo no gasta cuota ni tiempo indexando.

**14. Conclusiones (30 s)**
Lo conseguido y las mejoras posibles (re-ranking, búsqueda híbrida, router para preguntas fuera de dominio, evaluación automática). Cerrar con el enlace al repositorio.
