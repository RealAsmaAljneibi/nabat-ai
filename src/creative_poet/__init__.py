"""
src/creative_poet
==================
Worker 4 — Creative Composition Pipeline.

Sits alongside al_nassikh/ (Worker 1) and fatat_al_arab/ (Workers 2+3)
as a peer top-level package under src/.

  Al-Mulhim  (الملهِم)  — compositional scaffold generator    mode="scaffold"
  Al-Musharik (المشارك) — interactive verse co-author         mode="coauthor"
  Al-Hafiz   (الحافظ)   — deceased poet voice preservation    mode="preserve"
  Al-Muqayyim (المقيّم) — poem quality critic                 mode="critique"

Entry point: fatat_al_arab.orchestrator.run_creative(composition_context)
Personas:    creative_poet.personas   (MULHIM_PERSONA … MUQAYYIM_PERSONA)
State:       fatat_al_arab.state      (CompositionContext, CompositionState)
"""
