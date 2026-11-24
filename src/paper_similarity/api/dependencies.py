"""Connects the routes with the services."""

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session, sessionmaker

from paper_similarity.embeddings.base import Embedder
from paper_similarity.services.similarity import SimilarityService
from paper_similarity.services.submission import SubmissionService

# The objects in app.state are created once, when the application starts; see main.py.


def get_session_factory(request: Request) -> sessionmaker[Session]:
    session_factory: sessionmaker[Session] = request.app.state.session_factory
    return session_factory


def get_embedder(request: Request) -> Embedder:
    embedder: Embedder = request.app.state.embedder
    return embedder


SessionFactoryDependency = Annotated[sessionmaker[Session], Depends(get_session_factory)]
EmbedderDependency = Annotated[Embedder, Depends(get_embedder)]


def get_submission_service(
    embedder: EmbedderDependency, session_factory: SessionFactoryDependency
) -> SubmissionService:
    return SubmissionService(embedder, session_factory)


def get_similarity_service(session_factory: SessionFactoryDependency) -> SimilarityService:
    return SimilarityService(session_factory)


SubmissionServiceDependency = Annotated[SubmissionService, Depends(get_submission_service)]
SimilarityServiceDependency = Annotated[SimilarityService, Depends(get_similarity_service)]
