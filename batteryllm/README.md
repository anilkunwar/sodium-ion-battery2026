
# BatteryLLM

R1 : Static LLM dealing with six main scientific fields of Na-ion Battery

[![meaningtowords](https://img.shields.io/badge/naionbatteryllmcg-v1.0-green)](https://llm-guided-na-ion-battery-concept-graph1.streamlit.app/) (Static code that does not read the json database)

R2-R3 : The json file is read by the app

[![meaningtowords](https://img.shields.io/badge/naionbatteryllmcg-v2.0-yellow)](https://llm-guided-na-ion-battery-concept-graph2.streamlit.app/)

[![meaningtowords](https://img.shields.io/badge/naionbatteryllmcg-v3.0-yellow)](https://llm-guided-na-ion-battery-concept-graph3.streamlit.app/)

R4- R5 

[![meaningtowords](https://img.shields.io/badge/naionbatteryllmcg-v4.0-turquoise)](https://llm-guided-na-ion-battery-concept-graph4.streamlit.app/)

[![meaningtowords](https://img.shields.io/badge/naionbatteryllmcg-v5.0-turquoise)](https://llm-guided-na-ion-battery-concept-graph5.streamlit.app/)

R6-R7 (Full Graph First without Query and Sub-graph from Query to LLM)

[![meaningtowords](https://img.shields.io/badge/naionbatteryllmcg-v6.0-red)](https://llm-guided-na-ion-battery-concept-graph6.streamlit.app/) (Basic Version, The LLM learns from Concept Graph to Respond to the User's Query, The  subgraph is a pre-computed visualization tied to the problem category (e.g. anode_bottleneck), not dynamically constructed from the query content. This is why it appears generic and not distinct to the specific question about sodium vs lithium intercalation in graphite.)

[![meaningtowords](https://img.shields.io/badge/naionbatteryllmcg-v6.a-red)](https://llm-guided-na-ion-battery-concept-graph6a.streamlit.app/) (Basic Version, The LLM learns from Concept Graph to Respond to the User's Query, The  subgraph is a pre-computed visualization tied to the problem category (e.g. anode_bottleneck), not dynamically constructed from the query content. This is why it appears generic and not distinct to the specific question about sodium vs lithium intercalation in graphite.)

[![meaningtowords](https://img.shields.io/badge/naionbatteryllmcg-v7.0-red)](https://llm-guided-na-ion-battery-concept-graph7.streamlit.app/) (Advanced Version, The LLM learns from Concept Graph to Respond to the User's Query, OOM for LLM-Guided Q&A)


R8-R9 (Query Distilled Concept Graph Construction)

[![meaningtowords](https://img.shields.io/badge/naionbatteryllmcg-v8.0-blue)](https://llm-guided-na-ion-battery-concept-graph8.streamlit.app/) (Query-distilled option but does't take query at first, The LLM learns from Concept Graph to Respond to the User's Query, The  subgraph is a pre-computed visualization tied to the problem category (e.g. anode_bottleneck), not dynamically constructed from the query content. This is why it appears generic and not distinct to the specific question about sodium vs lithium intercalation in graphite.)

[![meaningtowords](https://img.shields.io/badge/naionbatteryllmcg-v8.a-blue)](https://llm-guided-na-ion-battery-concept-graph8a.streamlit.app/) (Static ontology mapping, Query-distilled option and takes query at first, The LLM learns from Concept Graph to Respond to the User's Query, The  subgraph is a pre-computed visualization tied to the problem category (e.g. anode_bottleneck), not dynamically constructed from the query content. This is why it appears generic and not distinct to the specific question about sodium vs lithium intercalation in graphite.)

[![meaningtowords](https://img.shields.io/badge/naionbatteryllmcg-v8.b-blue)](https://llm-guided-na-ion-battery-concept-graph8b.streamlit.app/) (Truncation problem-  iframe boundary clipping combined with unbounded tooltip content growth, Dynamic, query-conditioned topological and semantic re-weighting during ontology mapping procedure, Query-distilled option and takes query at first, The LLM learns from Concept Graph to Respond to the User's Query, The  subgraph is a pre-computed visualization tied to the problem category (e.g. anode_bottleneck), not dynamically constructed from the query content. This is why it appears generic and not distinct to the specific question about sodium vs lithium intercalation in graphite.)

[![meaningtowords](https://img.shields.io/badge/naionbatteryllmcg-v8.c-blue)](https://llm-guided-na-ion-battery-concept-graph8c.streamlit.app/) (Truncation problem-  iframe boundary clipping combined with unbounded tooltip content growth, Dynamic, query-conditioned topological and semantic re-weighting during ontology mapping procedure, Query-distilled option and takes query at first, The LLM learns from Concept Graph to Respond to the User's Query, The  subgraph is a pre-computed visualization tied to the problem category (e.g. anode_bottleneck), not dynamically constructed from the query content. This is why it appears generic and not distinct to the specific question about sodium vs lithium intercalation in graphite.)

[![meaningtowords](https://img.shields.io/badge/naionbatteryllmcg-v9.0-orange)](https://llm-guided-na-ion-battery-concept-graph9.streamlit.app/) (Query-distilled option and takes query at first, The LLM learns from Concept Graph to Respond to the User's Query, The  subgraph is a pre-computed visualization tied to the problem category (e.g. anode_bottleneck), not dynamically constructed from the query content. This is why it appears generic and not distinct to the specific question about sodium vs lithium intercalation in graphite.)

[![meaningtowords](https://img.shields.io/badge/naionbatteryllmcg-v10.0-blue)](https://llm-guided-na-ion-battery-concept-graph10.streamlit.app/) (Code 8c upgraded to prevent the truncation (node label truncation) of the titles of concepts in the "Concept Definition in Tooltips" Javascript html fronend [INCOMPLETE] , Dynamic, query-conditioned topological and semantic re-weighting during ontology mapping procedure, Query-distilled option and takes query at first, The LLM learns from Concept Graph to Respond to the User's Query, The  subgraph is a pre-computed visualization tied to the problem category (e.g. anode_bottleneck), not dynamically constructed from the query content. This is why it appears generic and not distinct to the specific question about sodium vs lithium intercalation in graphite.)

[![meaningtowords](https://img.shields.io/badge/naionbatteryllmcg-v10.a-blue)](https://llm-guided-na-ion-battery-concept-graph10a.streamlit.app/) (Code 8c upgraded to prevent the truncation (node label truncation) of the titles of concepts in the "Concept Definition in Tooltips" Javascript html fronend [COMPLETE], Dynamic, query-conditioned topological and semantic re-weighting during ontology mapping procedure, Query-distilled option and takes query at first, The LLM learns from Concept Graph to Respond to the User's Query, The  subgraph is a pre-computed visualization tied to the problem category (e.g. anode_bottleneck), not dynamically constructed from the query content. This is why it appears generic and not distinct to the specific question about sodium vs lithium intercalation in graphite.)


[![meaningtowords](https://img.shields.io/badge/naionbatteryllmcg-v100.0-blue)](https://llm-guided-na-ion-battery-concept-graph100.streamlit.app/) (Code 8c upgraded to prevent the truncation (node label truncation) of the titles of concepts in the "Concept Definition in Tooltips" Javascript html fronend [COMPLETE], Dynamic, query-conditioned topological and semantic re-weighting during ontology mapping procedure, Query-distilled option and takes query at first, The LLM learns from Concept Graph to Respond to the User's Query, The  subgraph is a pre-computed visualization tied to the problem category (e.g. anode_bottleneck), not dynamically constructed from the query content. This is why it appears generic and not distinct to the specific question about sodium vs lithium intercalation in graphite.)




