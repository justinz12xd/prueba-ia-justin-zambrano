from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy.orm import Session

from app.models.agent_session import AgentSession


class AgentSessionRepository:
    def __init__(self, db: Session):
        self.db = db

    def get(self, session_id: str) -> AgentSession | None:
        return self.db.get(AgentSession, session_id)

    def get_or_create(self, session_id: str | None, customer_id: int | None) -> AgentSession:
        if session_id:
            session = self.get(session_id)
            if session is not None:
                return session
        new_id = session_id or str(uuid.uuid4())
        session = AgentSession(session_id=new_id, customer_id=customer_id, conversation=[],
                                tokens_used=0)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def append_messages(self, session: AgentSession, messages: list[dict],
                         tokens_used_delta: int = 0) -> AgentSession:
        session.conversation = [*session.conversation, *messages]
        session.tokens_used += tokens_used_delta
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def close(self, session: AgentSession) -> AgentSession:
        session.ended_at = dt.datetime.now(dt.timezone.utc)
        self.db.commit()
        self.db.refresh(session)
        return session

    def delete(self, session: AgentSession) -> None:
        self.db.delete(session)
        self.db.commit()
