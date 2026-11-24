"""Connects the routes with the services."""

from typing import Annotated

from fastapi import Depends, Request

from paper_similarity.services.submission import SubmissionService


def get_submission_service(request: Request) -> SubmissionService:
    # Created once, when the application starts; see main.py.
    state = request.app.state
    return SubmissionService(state.embedder, state.session_factory)


SubmissionServiceDependency = Annotated[SubmissionService, Depends(get_submission_service)]
