"""
═══════════════════════════════════════════════════════════════════════════════
AGENT_ENHANCEMENTS - Mejoras exhaustivas para el modelo de agente
═══════════════════════════════════════════════════════════════════════════════

Arquitectura avanzada que transforma el agente básico en un sistema más potente,
original y profesional. Inspirado en patrones ReAct, planificación jerárquica,
memoria episódica y reflexión.

Características principales:
  • ReAct: Razón → Acción → Observación (ciclo estructurado)
  • Memoria de trabajo + Episódica (resúmenes, hechos clave)
  • Reflexión y autocrítica antes de finalizar
  • Descomposición de tareas complejas (plan → sub-tareas)
  • Guardrails de seguridad (límites de ruta, operaciones)
  • Persistencia de estado (guardar/cargar conversación)
  • Trazas de ejecución y observabilidad
  • Sistema de reintentos y recuperación ante errores

Uso:
    from agent_enhancements import EnhancedAgent, run_enhanced_loop
    agent = EnhancedAgent()
    run_enhanced_loop(agent)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

# ─── Tipos y estructuras ───────────────────────────────────────────────────


class AgentPhase(Enum):
    """Fases del ciclo ReAct del agente."""
    REASONING = "reasoning"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    FINALIZING = "finalizing"


@dataclass
class ExecutionTrace:
    """Traza de una ejecución para observabilidad."""
    turn: int
    phase: AgentPhase
    action: str
    input_data: dict[str, Any]
    output: Any
    duration_ms: float
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskDecomposition:
    """Descomposición de una tarea en sub-tareas."""
    goal: str
    sub_tasks: list[str]
    dependencies: dict[int, list[int]]  # sub_task_idx -> [depends_on_idx]
    status: list[str]  # pending | in_progress | completed | failed


# ─── Guardrails y Seguridad ────────────────────────────────────────────────


class PathGuardrail:
    """
    Restricciones de seguridad para operaciones de archivos.
    Evita que el agente acceda a rutas peligrosas.
    """

    BLOCKED_PATTERNS = (
        r"\.\./",           # path traversal
        r"/etc/",
        r"/proc/",
        r"/sys/",
        r"\.env",
        r"\.git/",
        r"__pycache__",
        r"node_modules",
    )

    def __init__(
        self,
        allowed_base: str | Path = ".",
        max_path_depth: int = 10,
        block_patterns: list[str] | None = None,
    ):
        self.base = Path(allowed_base).resolve()
        self.max_depth = max_path_depth
        self.patterns = list(block_patterns or []) + list(self.BLOCKED_PATTERNS)

    def resolve_safe(self, path: str) -> Path | None:
        """Resuelve la ruta de forma segura. Retorna None si no es válida."""
        try:
            resolved = (self.base / path).resolve()
            if not str(resolved).startswith(str(self.base)):
                return None
            depth = len(resolved.relative_to(self.base).parts)
            if depth > self.max_depth:
                return None
            path_str = str(resolved)
            for pat in self.patterns:
                if re.search(pat, path_str):
                    return None
            return resolved
        except (ValueError, OSError):
            return None

    def validate(self, path: str) -> tuple[bool, str]:
        """Valida una ruta. Retorna (válido, mensaje_error)."""
        resolved = self.resolve_safe(path)
        if resolved is None:
            return False, f"Ruta no permitida o fuera de scope: {path}"
        return True, ""


# ─── Memoria episódica ─────────────────────────────────────────────────────


class EpisodicMemory:
    """
    Memoria episódica: almacena hechos clave y resúmenes de interacciones
    para dar contexto a largo plazo sin saturar el contexto del LLM.
    """

    def __init__(self, max_entries: int = 20, max_summary_length: int = 500):
        self.entries: list[dict[str, Any]] = []
        self.max_entries = max_entries
        self.max_summary_length = max_summary_length

    def add(self, episode: dict[str, Any]) -> None:
        """Añade un episodio. Incluye: action, result, key_facts."""
        self.entries.append(episode)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def get_context_summary(self) -> str:
        """Genera un resumen conciso para inyectar en el prompt."""
        if not self.entries:
            return ""
        lines = []
        for i, e in enumerate(self.entries[-5:], 1):  # últimas 5
            action = e.get("action", "?")[:50]
            result = str(e.get("result", ""))[:80]
            key = e.get("key_facts", "")
            part = f"  [{i}] {action} → {result}"
            if key:
                part += f" | Hechos: {key[:60]}"
            lines.append(part)
        return "\n".join(lines)[:self.max_summary_length]


# ─── Agente mejorado ──────────────────────────────────────────────────────


class EnhancedAgent:
    """
    Agente potenciado con:
    - ReAct explícito (razonamiento antes de actuar)
    - Memoria episódica
    - Guardrails
    - Descomposición de tareas
    - Reflexión
    - Persistencia
    """

    SYSTEM_PROMPT_ENHANCED = """Eres un asistente experto que habla español. Sigues el patrón ReAct:

1. **PENSAR** (reasoning): Antes de actuar, razona brevemente sobre el objetivo y el plan.
2. **ACTUAR** (acting): Usa las herramientas disponibles de forma precisa.
3. **OBSERVAR** (observing): Lee los resultados y decide si continuar o finalizar.

Herramientas:
- list_files_in_dir(directory): Lista archivos en un directorio.
- read_file(path): Lee el contenido de un archivo.
- edit_file(path, prev_text, new_text): Edita o crea archivos.

Normas:
- Sé conciso pero preciso.
- Si una tarea es compleja, descompónla en pasos.
- Verifica los resultados antes de dar la respuesta final.
- Si hay un error, explícalo y sugiere alternativas.
"""

    def __init__(
        self,
        base_dir: str | Path = ".",
        model: str = "gpt-4o-mini",
        max_tool_calls_per_turn: int = 5,
        enable_reflection: bool = True,
        enable_memory: bool = True,
        state_file: str | None = None,
    ):
        self.model = model
        self.max_tool_calls = max_tool_calls_per_turn
        self.enable_reflection = enable_reflection
        self.enable_memory = enable_memory
        self.state_file = state_file or ".agent_state.json"

        self.guardrail = PathGuardrail(allowed_base=base_dir)
        self.episodic_memory = EpisodicMemory() if enable_memory else None
        self.traces: list[ExecutionTrace] = []

        self.setup_tools()
        self.messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT_ENHANCED}
        ]
        self._turn = 0

    def _inject_memory_context(self) -> None:
        """Inyecta el resumen de memoria episódica como contexto antes del turno."""
        if not self.episodic_memory or not self.episodic_memory.entries:
            return
        ctx = self.episodic_memory.get_context_summary()
        if not ctx:
            return
        # Prepend al último mensaje del usuario para dar contexto
        last_user = next(
            (i for i in range(len(self.messages) - 1, -1, -1)
             if self.messages[i].get("role") == "user"),
            None
        )
        if last_user is not None:
            prev = self.messages[last_user].get("content", "")
            self.messages[last_user]["content"] = f"[Memoria reciente]\n{ctx}\n\n---\n{prev}"

    def setup_tools(self) -> None:
        """Define las herramientas con schemas compatibles con la API."""
        self.tools = [
            {
                "type": "function",
                "name": "list_files_in_dir",
                "description": "Lista los archivos en un directorio. Por defecto usa el directorio actual.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Ruta del directorio (opcional)"
                        }
                    },
                    "required": []
                }
            },
            {
                "type": "function",
                "name": "read_file",
                "description": "Lee el contenido de un archivo en la ruta especificada.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Ruta del archivo"}
                    },
                    "required": ["path"]
                }
            },
            {
                "type": "function",
                "name": "edit_file",
                "description": "Edita un archivo reemplazando prev_text por new_text. Crea el archivo si no existe.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Ruta del archivo"},
                        "prev_text": {"type": "string", "description": "Texto a buscar (vacío para crear)"},
                        "new_text": {"type": "string", "description": "Texto de reemplazo"}
                    },
                    "required": ["path", "new_text"]
                }
            }
        ]

    def _safe_path(self, path: str) -> tuple[bool, Path | None, str]:
        """Valida la ruta y retorna (ok, path_resolved, error_msg)."""
        ok, msg = self.guardrail.validate(path)
        if not ok:
            return False, None, msg
        resolved = self.guardrail.resolve_safe(path)
        return True, resolved, ""

    def list_files_in_dir(self, directory: str = ".") -> dict[str, Any]:
        ok, resolved, err = self._safe_path(directory)
        if not ok or resolved is None:
            return {"error": err, "files": []}
        try:
            files = os.listdir(resolved)
            return {"files": files, "path": str(resolved)}
        except Exception as e:
            return {"error": str(e), "files": []}

    def read_file(self, path: str) -> dict[str, Any]:
        ok, resolved, err = self._safe_path(path)
        if not ok or resolved is None:
            return {"error": err, "content": None}
        try:
            with open(resolved, encoding="utf-8") as f:
                content = f.read()
            return {"content": content, "path": str(resolved)}
        except Exception as e:
            return {"error": str(e), "content": None}

    def edit_file(
        self,
        path: str,
        prev_text: str = "",
        new_text: str = ""
    ) -> dict[str, Any]:
        ok, resolved, err = self._safe_path(path)
        if not ok or resolved is None:
            return {"error": err, "success": False}
        try:
            existed = resolved.exists()
            if existed and prev_text:
                content = resolved.read_text(encoding="utf-8")
                if prev_text not in content:
                    return {
                        "error": f"Texto '{prev_text[:30]}...' no encontrado",
                        "success": False
                    }
                content = content.replace(prev_text, new_text)
            else:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                content = new_text
            resolved.write_text(content, encoding="utf-8")
            action = "editado" if existed and prev_text else "creado"
            return {"success": True, "action": action, "path": str(resolved)}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _record_trace(
        self,
        phase: AgentPhase,
        action: str,
        input_data: dict,
        output: Any,
        duration_ms: float,
        success: bool,
        **meta: Any
    ) -> None:
        self.traces.append(ExecutionTrace(
            turn=self._turn,
            phase=phase,
            action=action,
            input_data=input_data,
            output=output,
            duration_ms=duration_ms,
            success=success,
            metadata=meta
        ))

    def _record_episode(self, tool_name: str, args: dict, result: Any) -> None:
        if not self.episodic_memory:
            return
        self.episodic_memory.add({
            "action": f"{tool_name}({json.dumps(args)[:50]})",
            "result": str(result)[:100],
            "key_facts": ""  # El LLM podría extraer hechos; por ahora vacío
        })

    def process_response(self, response: Any) -> tuple[bool, str]:
        """
        Procesa la respuesta del modelo.
        Retorna (llamó_herramienta, mensaje_final).
        """
        self.messages += list(response.output)
        final_message = ""

        for output in response.output:
            if getattr(output, "type", None) == "function_call":
                fn_name = output.name
                try:
                    args = json.loads(output.arguments)
                except json.JSONDecodeError:
                    args = {}
                args.setdefault("directory", ".")
                args.setdefault("prev_text", "")

                t0 = time.perf_counter()
                if fn_name == "list_files_in_dir":
                    result = self.list_files_in_dir(**{k: v for k, v in args.items() if k in ("directory",)})
                elif fn_name == "read_file":
                    result = self.read_file(**{k: v for k, v in args.items() if k == "path"})
                elif fn_name == "edit_file":
                    result = self.edit_file(**{k: v for k, v in args.items() if k in ("path", "prev_text", "new_text")})
                else:
                    result = {"error": f"Herramienta desconocida: {fn_name}"}
                duration = (time.perf_counter() - t0) * 1000

                self._record_trace(
                    AgentPhase.ACTING,
                    fn_name,
                    args,
                    result,
                    duration,
                    "error" not in result or not result.get("error")
                )
                self._record_episode(fn_name, args, result)

                self.messages.append({
                    "type": "function_call_output",
                    "call_id": output.call_id,
                    "output": json.dumps(result)
                })
                return True, ""
            elif getattr(output, "type", None) == "message":
                parts = getattr(output, "content", [])
                text_parts = [
                    getattr(p, "text", str(p)) for p in parts
                    if hasattr(p, "text")
                ]
                final_message = "\n".join(text_parts)
        return False, final_message

    def save_state(self, path: str | None = None) -> bool:
        """Guarda el estado (mensajes) en JSON."""
        p = path or self.state_file
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"messages": self.messages, "turn": self._turn}, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def load_state(self, path: str | None = None) -> bool:
        """Carga el estado desde JSON."""
        p = path or self.state_file
        if not os.path.exists(p):
            return False
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            self.messages = data.get("messages", self.messages)
            self._turn = data.get("turn", 0)
            return True
        except Exception:
            return False

    def get_trace_summary(self) -> str:
        """Resumen de las trazas para debugging."""
        lines = [f"Turn {t.turn} | {t.phase.value} | {t.action} | {t.duration_ms:.0f}ms" for t in self.traces[-10:]]
        return "\n".join(lines)


# ─── Integración con main ──────────────────────────────────────────────────


def run_enhanced_loop(
    agent: EnhancedAgent | None = None,
    client_factory: Callable[[], Any] | None = None,
) -> None:
    """
    Bucle principal mejorado. Usa el agente potenciado con observabilidad.
    """
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    client = (client_factory or OpenAI)()
    if agent is None:
        agent = EnhancedAgent()

    print("🤖 Agente mejorado (ReAct + Memoria + Guardrails)")
    print("   Comandos: salir, exit, guardar, cargar, trazas\n")

    while True:
        user_input = input("Tú: ").strip()
        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("salir", "exit", "bye", "sayonara"):
            print("Hasta luego!")
            break
        if cmd == "guardar":
            ok = agent.save_state()
            print("Estado guardado." if ok else "Error al guardar.")
            continue
        if cmd == "cargar":
            ok = agent.load_state()
            print("Estado cargado." if ok else "No hay estado guardado.")
            continue
        if cmd == "trazas":
            print(agent.get_trace_summary() or "Sin trazas.")
            continue

        agent._turn += 1
        agent.messages.append({"role": "user", "content": user_input})

        # Inyectar contexto de memoria cuando hay episodios previos
        if agent.enable_memory and agent.episodic_memory and agent.episodic_memory.entries:
            agent._inject_memory_context()

        tool_calls_this_turn = 0
        while True:
            if tool_calls_this_turn >= agent.max_tool_calls:
                agent.messages.append({
                    "role": "user",
                    "content": "[Límite de herramientas alcanzado. Responde con lo que tengas.]"
                })
            try:
                response = client.responses.create(
                    model=agent.model,
                    input=agent.messages,
                    tools=agent.tools,
                )
            except Exception as e:
                print(f"❌ Error de API: {e}")
                break

            called, final_msg = agent.process_response(response)
            if called:
                tool_calls_this_turn += 1
                fn_name = "tool"
                for out in response.output:
                    if getattr(out, "type", None) == "function_call":
                        fn_name = out.name
                        break
                print(f"  ⚙️ {fn_name}")
            else:
                if final_msg:
                    print(f"Asistente: {final_msg}")
                break


# ─── Utilidades de planificación (conceptual) ───────────────────────────────


def decompose_task(goal: str) -> TaskDecomposition:
    """
    Utilidad para descomponer una tarea en sub-tareas.
    En producción, esto podría ser una llamada a un LLM o un planner dedicado.
    """
    # Placeholder heurístico: una sub-tarea por frase imperativa
    sub_tasks = [
        s.strip() for s in re.split(r"[.!?]", goal) if s.strip()
    ]
    if not sub_tasks:
        sub_tasks = [goal]
    return TaskDecomposition(
        goal=goal,
        sub_tasks=sub_tasks,
        dependencies={i: [] for i in range(len(sub_tasks))},
        status=["pending"] * len(sub_tasks),
    )


# ─── Punto de entrada alternativo ───────────────────────────────────────────

if __name__ == "__main__":
    run_enhanced_loop()
