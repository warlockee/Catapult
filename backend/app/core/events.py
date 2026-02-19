"""
Domain event system for loose coupling between services.

This module provides a simple in-process event dispatcher that allows
services to communicate without direct dependencies. Events are dispatched
synchronously within the same request context.

For async/background processing, handlers can enqueue Celery tasks.
"""
import logging
from typing import Callable, Dict, List, Any, Type
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

logger = logging.getLogger(__name__)


# =============================================================================
# Base Event Classes
# =============================================================================

@dataclass
class DomainEvent:
    """Base class for all domain events."""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def event_type(self) -> str:
        """Return the event type name."""
        return self.__class__.__name__


# =============================================================================
# Release Events
# =============================================================================

@dataclass
class ReleaseCreatedEvent(DomainEvent):
    """Emitted when a new release is created."""
    release_id: UUID = None
    model_id: UUID = None
    model_name: str = None
    version: str = None
    auto_build: bool = False
    build_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReleaseUpdatedEvent(DomainEvent):
    """Emitted when a release is updated."""
    release_id: UUID = None
    changes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReleaseDeletedEvent(DomainEvent):
    """Emitted when a release is deleted."""
    release_id: UUID = None
    model_id: UUID = None


# =============================================================================
# Docker Build Events
# =============================================================================

@dataclass
class DockerBuildRequestedEvent(DomainEvent):
    """Emitted when a Docker build is requested."""
    release_id: UUID = None
    image_tag: str = None
    build_type: str = None
    artifact_ids: List[UUID] = field(default_factory=list)
    dockerfile_content: str = None


@dataclass
class DockerBuildStartedEvent(DomainEvent):
    """Emitted when a Docker build starts."""
    build_id: UUID = None
    release_id: UUID = None
    image_tag: str = None


@dataclass
class DockerBuildCompletedEvent(DomainEvent):
    """Emitted when a Docker build completes successfully."""
    build_id: UUID = None
    release_id: UUID = None
    image_tag: str = None


@dataclass
class DockerBuildFailedEvent(DomainEvent):
    """Emitted when a Docker build fails."""
    build_id: UUID = None
    release_id: UUID = None
    error_message: str = None


# =============================================================================
# Deployment Events
# =============================================================================

@dataclass
class DeploymentCreatedEvent(DomainEvent):
    """Emitted when a deployment is created."""
    deployment_id: UUID = None
    release_id: UUID = None
    environment: str = None


@dataclass
class DeploymentStatusChangedEvent(DomainEvent):
    """Emitted when a deployment status changes."""
    deployment_id: UUID = None
    old_status: str = None
    new_status: str = None


# =============================================================================
# Event Dispatcher
# =============================================================================

class EventDispatcher:
    """
    Simple in-process event dispatcher.

    Handlers are registered per event type and called synchronously
    when events are dispatched.
    """

    def __init__(self):
        self._handlers: Dict[Type[DomainEvent], List[Callable]] = {}

    def register(
        self,
        event_type: Type[DomainEvent],
        handler: Callable[[DomainEvent], Any],
    ) -> None:
        """
        Register a handler for an event type.

        Args:
            event_type: The event class to handle
            handler: Callable that takes the event as argument
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Registered handler {handler.__name__} for {event_type.__name__}")

    def unregister(
        self,
        event_type: Type[DomainEvent],
        handler: Callable[[DomainEvent], Any],
    ) -> None:
        """
        Unregister a handler for an event type.

        Args:
            event_type: The event class
            handler: The handler to remove
        """
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

    def dispatch(self, event: DomainEvent) -> None:
        """
        Dispatch an event to all registered handlers.

        Args:
            event: The event to dispatch
        """
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        logger.debug(f"Dispatching {event_type.__name__} to {len(handlers)} handlers")

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Handler {handler.__name__} failed for {event_type.__name__}: {e}"
                )

    async def dispatch_async(self, event: DomainEvent) -> None:
        """
        Dispatch an event to all registered handlers (async version).

        Handlers can be either sync or async functions.

        Args:
            event: The event to dispatch
        """
        import asyncio

        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        logger.debug(f"Dispatching {event_type.__name__} to {len(handlers)} handlers (async)")

        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(
                    f"Handler {handler.__name__} failed for {event_type.__name__}: {e}"
                )

    def clear(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()


# Global event dispatcher instance
event_dispatcher = EventDispatcher()


# =============================================================================
# Decorator for registering handlers
# =============================================================================

def handles(event_type: Type[DomainEvent]):
    """
    Decorator to register a function as an event handler.

    Example:
        @handles(ReleaseCreatedEvent)
        def on_release_created(event: ReleaseCreatedEvent):
            if event.auto_build:
                trigger_docker_build(event)
    """
    def decorator(func: Callable):
        event_dispatcher.register(event_type, func)
        return func
    return decorator
