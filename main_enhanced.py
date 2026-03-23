"""
Punto de entrada para el agente mejorado.
Ejecuta: python main_enhanced.py

Nota: Si tu proyecto usa gpt-5-nano, cambia model="gpt-5-nano"
"""
from agent_enhancements import EnhancedAgent, run_enhanced_loop

if __name__ == "__main__":
    agent = EnhancedAgent(
        model="gpt-4o-mini",
        enable_reflection=True,
        enable_memory=True,
        max_tool_calls_per_turn=8,
    )
    run_enhanced_loop(agent)
