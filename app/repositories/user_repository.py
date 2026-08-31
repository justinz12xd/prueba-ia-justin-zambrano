from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.user import UserAccount


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_email(self, email: str) -> UserAccount | None:
        stmt = select(UserAccount).where(UserAccount.email == email, UserAccount.is_active.is_(True))
        return self.db.scalars(stmt).first()

    def create(self, email: str, hashed_password: str, role: str,
               customer_id: int | None = None) -> UserAccount:
        user = UserAccount(email=email, hashed_password=hashed_password, role=role,
                            customer_id=customer_id)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
