"""
/api/conversations  — CRUD for conversation history.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models.database import ConversationModel, MessageModel
from app.models.schemas import ConversationCreate, ConversationOut
from database.connection import get_db

router = APIRouter(prefix="/api", tags=["conversations"])


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(db: Session = Depends(get_db)):
    return db.query(ConversationModel).order_by(ConversationModel.created_at.desc()).all()


@router.post("/conversations", response_model=ConversationOut, status_code=201)
def create_conversation(payload: ConversationCreate, db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)
    convo = ConversationModel(
        id=str(uuid.uuid4()),
        session_id=payload.session_id,
        created_at=now,
        updated_at=now,
    )
    db.add(convo)
    db.flush()   # get the id before adding messages

    for msg_in in payload.messages:
        msg = MessageModel(
            id=str(uuid.uuid4()),
            conversation_id=convo.id,
            content=msg_in.content,
            source=msg_in.source,
            language=msg_in.language,
            confidence=msg_in.confidence,
            timestamp=now,
        )
        db.add(msg)

    db.commit()
    db.refresh(convo)
    return convo


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    convo = db.get(ConversationModel, conversation_id)
    if convo is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    db.delete(convo)
    db.commit()
