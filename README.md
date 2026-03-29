
## Configuración
Para ejecutar el proyecto es necesario:
- Opcional: Crea un ambiente virtual
- Instala las dependencias ejecutando 
	- ```  pip install -r requirements.txt ```
- Crea un archivo llamado ```.env```
	- En el archivo agrega tu llave de OpenAI:
	- ```OPENAI_API_KEY=XXXXXX```

## Ejecución
- Activar el ambiente virtual
- Ejecutar ```main.py```

## Agente mejorado

El archivo `agent_enhancements.py` añade mejoras exhaustivas al agente original:

| Característica | Descripción |
|----------------|-------------|
| **ReAct** | Patrón Razón → Acción → Observación |
| **Memoria episódica** | Resúmenes de interacciones para contexto a largo plazo |
| **Guardrails** | Restricciones de rutas (evita `.env`, `.git`, etc.) |
| **Persistencia** | Comandos `guardar` / `cargar` para guardar estado |
| **Trazas** | Comando `trazas` para ver historial de ejecución |
| **Descomposición de tareas** | Utilidad para tareas complejas |

**Ejecutar agente mejorado:**
```bash
python main_enhanced.py
```

## Documentación exhaustiva

En `docs/DOCUMENTACION.md` encontrarás:

- **Parte I — Teoría básica:** Qué es un agente, fundamentos de LLMs, tool-calling, ReAct, memoria, guardrails, persistencia y observabilidad.
- **Parte II — Implementación:** Estructura del código, flujos detallados, cómo añadir herramientas y personalizar el agente.
- **Glosario y referencias** para profundizar.

La documentación está pensada para principiantes e incluye explicaciones paso a paso de cada concepto.

---
Puede ser conveniente usar codex 5.3

