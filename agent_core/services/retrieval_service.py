from agent_core.rag_retriever import retrieve_tf_knowledge


class RetrievalService:
    def retrieve(self, *, user_query: str, target_resources: list[str]) -> str:
        return retrieve_tf_knowledge(
            user_query=user_query,
            target_resources=target_resources,
        )
