"""
Seeds Arize Phoenix with a Curated BACEN RAG Golden Set Dataset.
Used for offline, asynchronous RAG experiments and evaluations.
"""

import logging
import os
from phoenix.client import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed_dataset")

# Golden Set Data: BACEN Questions -> Ground Truth BACEN Regulation Chunks
GOLDEN_SET_EXAMPLES = [
    {
        "input": "Qual é o limite padrão para transferências PIX durante a noite?",
        "expected_resolution": "BCB-142",
        "expected_text": "De acordo com a Resolução BCB nº 142, estabelece-se o limite padrão de R$ 1.000,00 (100.000 centavos) para transações PIX realizadas por pessoas físicas no horário noturno compreendido entre 20:00 e 06:00.",
    },
    {
        "input": "Em quanto tempo a alteração do limite do PIX noturno é aprovada?",
        "expected_resolution": "BCB-142",
        "expected_text": "Clientes podem solicitar alteração do limite com prazo de aprovação de 24h a 48h.",
    },
    {
        "input": "Como funciona o bloqueio de valores em caso de suspeita de golpe no PIX?",
        "expected_resolution": "BCB-103",
        "expected_text": "A Resolução BCB nº 103 disciplina o Mecanismo Especial de Devolução (MED). Em casos de fundada suspeita de fraude, golpe ou engenharia social, a instituição financeira receptora deve efetuar o bloqueio cautelar imediato dos valores por até 72 horas.",
    },
    {
        "input": "O que é o Mecanismo Especial de Devolução (MED)?",
        "expected_resolution": "BCB-103",
        "expected_text": "Mecanismo Especial de Devolução (MED) para análise de estorno de valores em casos de fraude ou engenharia social.",
    },
    {
        "input": "Quais são as chaves PIX válidas para autenticação no DICT?",
        "expected_resolution": "BCB-001",
        "expected_text": "Todas as transferências exigem validação prévia de chave ativa (CPF, CNPJ, e-mail, telefone ou chave aleatória).",
    },
    {
        "input": "Transferência para chave PIX bloqueada por ordem judicial é permitida?",
        "expected_resolution": "BCB-001",
        "expected_text": "É proibida a conclusão de transferências para chaves sinalizadas como bloqueadas por ordem judicial ou administrativa.",
    },
]


def seed_phoenix_dataset():
    phoenix_url = os.getenv(
        "PHOENIX_COLLECTOR_HTTP_ENDPOINT",
        "http://arize-phoenix-service.guardrails.svc.cluster.local:6006",
    )
    logger.info(f"Connecting to Arize Phoenix at '{phoenix_url}'...")

    client = Client(endpoint=phoenix_url)

    dataset_name = "bacen-rag-golden-set"
    logger.info(f"Creating/updating Phoenix dataset: '{dataset_name}'...")

    dataset = client.upload_dataset(
        dataset_name=dataset_name,
        inputs=[{"input": item["input"]} for item in GOLDEN_SET_EXAMPLES],
        outputs=[
            {
                "expected_resolution": item["expected_resolution"],
                "expected_text": item["expected_text"],
            }
            for item in GOLDEN_SET_EXAMPLES
        ],
    )

    logger.info(
        f"✅ Successfully seeded '{dataset_name}' with {len(GOLDEN_SET_EXAMPLES)} BACEN ground-truth examples."
    )
    return dataset


if __name__ == "__main__":
    seed_phoenix_dataset()
