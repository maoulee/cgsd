"""Environment-owned routes and resource stages; no model fact declarations.

Node keys are canonical graph IDs supplied by the adapter. A route records
independent observations, not global first-owner credit. State proposals are
read-only; actual execution activates stages and spends resources.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import copy


class RouteBoundary(ValueError):
    pass


@dataclass
class Stage:
    stage_id: str
    domain: set[str]
    activated: bool = False
    sg_used: int = 0
    rr_used: int = 0
    tried_relations: list[str] = field(default_factory=list)
    source_calls: tuple[str, ...] = ()


@dataclass
class Route:
    route_id: str
    root: str
    nodes: set[str] = field(default_factory=set)
    facts: set[tuple[str, str, str]] = field(default_factory=set)
    frontier: set[str] = field(default_factory=set)
    stages: list[Stage] = field(default_factory=list)
    version: int = 0
    last_delivery: int = 0


@dataclass(frozen=True)
class Ticket:
    route_id: str
    root: str
    centers: frozenset[str]
    domain: frozenset[str]
    stage_id: str | None
    source_calls: tuple[str, ...] = ()
    new_route: bool = False


class RouteStore:
    def __init__(self, center_limit=3, shift_limit=3, rr_limit=3):
        self.center_limit = center_limit
        self.shift_limit = shift_limit
        self.rr_limit = rr_limit
        self.routes: dict[str, Route] = {}
        self.roots: dict[str, str] = {}
        self.question_roots: set[str] = set()
        self.active: str | None = None
        self.events: list[dict] = []
        self.background: list[dict] = []
        self.cache: dict[str, dict] = {}
        self.plan_args: dict | None = None

    def seed(self, roots, question_roots=()):
        roots = list(dict.fromkeys(roots))
        if not self.routes:
            self.question_roots.update(roots)
        self.question_roots.update(question_roots)
        for root in roots:
            if root not in self.roots:
                if root not in self.question_roots and any(root in r.nodes for r in self.routes.values()):
                    # A later plan cannot relabel an already connected
                    # intermediate center as a fresh independent budget.
                    continue
                rid = f"r{len(self.routes) + 1}"
                self.routes[rid] = Route(rid, root, nodes={root})
                self.roots[root] = rid

    def groups(self, centers, preferred=None):
        """Return independent groups without consuming a resource or stage.

        Question roots retain their independent view even if another route
        happens to deliver that root. Shared descendants prefer the active
        graph, then the most recently delivered matching graph. No global KG
        shortest path is consulted and existing routes are never merged.
        """
        groups = {}
        next_new = len(self.routes) + 1
        for node in sorted(set(centers)):
            # A returned frontier reference carries provenance even when
            # the same entity belongs to more than one independent route.
            explicit = (preferred or {}).get(node, ())
            if explicit:
                for rid in sorted(explicit):
                    groups.setdefault(rid, set()).add(node)
                continue
            rid = self.roots.get(node)
            if rid is None and self.active in self.routes:
                if node in self.routes[self.active].nodes:
                    rid = self.active
            if rid is None:
                matches = [r for r in self.routes.values() if node in r.nodes]
                if matches:
                    rid = max(matches, key=lambda r: (r.last_delivery, r.route_id)).route_id
            if rid is None:
                rid = f"r{next_new}"
                next_new += 1
            groups.setdefault(rid, set()).add(node)
        return groups

    def propose(self, route_id, centers, tool, check_budget=True):
        centers = frozenset(centers)
        if not centers:
            raise RouteBoundary("CENTER: choose a question anchor or delivered center.")
        route = self.routes.get(route_id)
        if route is None:
            ticket = Ticket(route_id, min(centers), centers, centers, None, new_route=True)
        else:
            stage = next((s for s in reversed(route.stages) if centers <= s.domain), None)
            sources, domain = list(stage.source_calls) if stage else [], set(centers)
            if stage is None:
                # One complete frontier is one stage, including subset calls.
                for event in reversed(self.events):
                    if event.get("route_id") != route_id or event.get("outcome") != "delivered":
                        continue
                    frontier = set(event.get("frontier", []))
                    if centers <= frontier:
                        domain = frontier
                        sources = [event["call_id"]]
                        break
            ticket = Ticket(route_id, route.root, centers, frozenset(domain),
                            stage.stage_id if stage else None, tuple(sources))
        if check_budget:
            self.check(ticket, tool)
        return ticket

    def check(self, ticket, tool):
        route = self.routes.get(ticket.route_id)
        stage = next((s for s in route.stages if s.stage_id == ticket.stage_id), None) if route else None
        if stage is None and route:
            stage = next((s for s in reversed(route.stages) if ticket.centers <= s.domain), None)
        if tool == "retrieve_subgraph":
            if stage and stage.sg_used >= self.center_limit:
                raise RouteBoundary(f"CENTER_BUDGET: {ticket.route_id}/{stage.stage_id} used {stage.sg_used}/{self.center_limit} SG attempts — this call did not execute."
                                    "\nadvice: existing evidence remains valid; infer and answer from the delivered records, or use another lane only for a still-unexecuted step.")
            active_count = sum(s.activated for s in route.stages) if route else 0
            if not (stage and stage.activated) and active_count >= self.shift_limit + 1:
                raise RouteBoundary(f"ROUTE_BUDGET: {ticket.route_id} already used {self.shift_limit} center changes — this call did not execute."
                                    "\nadvice: infer and answer from the delivered records, or revisit a stage with remaining attempts / switch routes only for a still-unexecuted step.")
        elif tool == "retrieve_relations":
            if stage and stage.sg_used >= self.center_limit:
                raise RouteBoundary(f"CENTER_BUDGET: {ticket.route_id}/{stage.stage_id} has no SG attempts left — this relation search did not execute."
                                    "\nadvice: infer and answer from the delivered records, or use a supported successor / another route only for a still-unexecuted step.")
            active_count = sum(s.activated for s in route.stages) if route else 0
            if not (stage and stage.activated) and active_count >= self.shift_limit + 1:
                raise RouteBoundary(f"ROUTE_BUDGET: {ticket.route_id} has no new center stages left — this call did not execute."
                                    "\nadvice: infer and answer from the delivered records, or revisit an existing stage with budget / switch routes only for a still-unexecuted step.")
            if stage and stage.rr_used >= self.rr_limit:
                raise RouteBoundary(f"RELATION_BUDGET: {ticket.route_id}/{stage.stage_id} used {stage.rr_used}/{self.rr_limit} relation searches — this call did not execute."
                                    "\nadvice: infer and answer from the delivered records, or select from the returned relations / switch routes only for a still-unexecuted step.")

    def begin(self, ticket, tool, relations=()):
        self.check(ticket, tool)
        previous = self.active
        route = self.routes.get(ticket.route_id)
        if route is None:
            route = Route(ticket.route_id, ticket.root, nodes={ticket.root})
            self.routes[route.route_id] = route
            self.roots.setdefault(ticket.root, route.route_id)
        stage = next((s for s in route.stages if s.stage_id == ticket.stage_id), None)
        if stage is None:
            # Multiple prepared calls may propose the same new stage.
            stage = next((s for s in reversed(route.stages) if ticket.centers <= s.domain), None)
        if stage is None:
            stage = Stage(f"s{len(route.stages) + 1}", set(ticket.domain), source_calls=ticket.source_calls)
            route.stages.append(stage)
        if tool == "retrieve_subgraph":
            stage.activated = True
            stage.sg_used += 1
            stage.tried_relations = list(dict.fromkeys(stage.tried_relations + list(relations)))
        elif tool == "retrieve_relations":
            stage.rr_used += 1
        self.active = route.route_id
        event = {"call_id": f"c{len(self.events) + 1}", "route_id": route.route_id,
                 "stage_id": stage.stage_id, "tool": tool, "centers": sorted(ticket.centers),
                 "source_calls": list(ticket.source_calls), "executed": True,
                 "outcome": "pending", "facts": [], "frontier": [],
                 "query_relations": list(relations), "result": "",
                 "transition": ("start_route" if previous is None or ticket.new_route else
                                "switch_route" if previous != route.route_id else "continue_route")}
        self.events.append(event)
        return event

    def finish(self, event, facts=(), frontier=(), result="", delivered=False, attributes=False):
        route = self.routes[event["route_id"]]
        facts = {tuple(f) for f in facts}
        if delivered:
            # Every admitted edge must attach to the actual input or existing
            # route. A disjoint render appendix is not silently joined in.
            connected = set(route.nodes) | set(event["centers"])
            pending = set(facts)
            while pending:
                attached = {f for f in pending if f[0] in connected or f[2] in connected}
                if not attached:
                    raise RouteBoundary("UNATTACHED_EVIDENCE: delivered ledger contains a component disconnected from this route; inspect the materializer.")
                for head, rel, tail in attached:
                    connected.update((head, tail))
                pending -= attached
            delta = facts - route.facts
            old_front = set(route.frontier)
            route.facts |= facts
            route.nodes = connected
            route.frontier = set(frontier) & connected
            if delta:
                route.version += 1
            route.last_delivery = len(self.events)
            event.update(outcome="delivered", facts=[list(f) for f in sorted(facts)],
                         frontier=sorted(route.frontier), new_fact_count=len(delta),
                         new_frontier_nodes=sorted(route.frontier - old_front),
                         attribute_evidence_added=bool(delta and attributes),
                         progress=("attribute_evidence" if delta and attributes else
                                   "frontier_evidence" if delta and route.frontier - old_front else
                                   "evidence_added" if delta else "no_information_gain"))
        else:
            event.update(outcome="no_evidence" if event["tool"] == "retrieve_subgraph" else "relations",
                         progress="no_information_gain" if event["tool"] == "retrieve_subgraph" else "relation_search",
                         new_fact_count=0, new_frontier_nodes=[], attribute_evidence_added=False)
        event["result"] = result

    def reuse(self, ticket, tool, result, cached_call):
        # Cache results carry no fresh walk and do not claim delivered facts.
        event = {"call_id": f"c{len(self.events) + 1}", "route_id": ticket.route_id,
                 "stage_id": ticket.stage_id, "tool": tool, "centers": sorted(ticket.centers),
                 "source_calls": [cached_call], "executed": False, "outcome": "cached",
                 "progress": "no_information_gain", "facts": [], "frontier": [],
                 "result": result}
        self.events.append(event)
        self.active = ticket.route_id
        return event

    def status(self, route_id):
        route = self.routes.get(route_id)
        if route is None:
            return "new route; initial center has 3 SG attempts"
        active_count = sum(s.activated for s in route.stages)
        stages = "; ".join(f"{s.stage_id}: SG {s.sg_used}/{self.center_limit}, RR {s.rr_used}/{self.rr_limit}" for s in route.stages)
        return f"{route_id}: center changes {max(0, active_count - 1)}/{self.shift_limit}; {stages or 'initial stage not executed'}"

    def to_dict(self):
        routes = []
        for route in self.routes.values():
            routes.append({"route_id": route.route_id, "root": route.root,
                           "nodes": sorted(route.nodes), "facts": [list(f) for f in sorted(route.facts)],
                           "frontier": sorted(route.frontier), "version": route.version,
                           "stages": [{**asdict(s), "domain": sorted(s.domain)} for s in route.stages]})
        return {"schema": "seq_routes_v1", "active": self.active,
                "question_roots": sorted(self.question_roots),
                "routes": routes, "events": copy.deepcopy(self.events),
                "background": copy.deepcopy(self.background)}


def credit_messages(sidecar, include_routes=None, mask_routes=(), include_background=False):
    """Tools and results only; whole-route masks; never copy thoughts/answers.

    An independently delivered shared fact remains in a surviving route's
    original result. Plan/path reminders are excluded by default; explicitly
    opt into fixed background when evaluating that different conditioning.
    This is a prompt constructor, not a probability/credit estimator.
    """
    messages = copy.deepcopy(sidecar.get("background", [])) if include_background else []
    allowed = set(include_routes) if include_routes is not None else None
    masked = set(mask_routes)
    for event in sidecar.get("events", []):
        rid = event.get("route_id")
        if rid in masked or (allowed is not None and rid not in allowed):
            continue
        if event.get("tool") not in ("retrieve_relations", "retrieve_subgraph"):
            continue
        # Stored content is canonical action, never the sampled response.
        action = event.get("action")
        if not action:
            continue
        messages.append({"role": "assistant", "content": action})
        messages.append({"role": "tool", "name": event["tool"], "content": event.get("result", "")})
    return messages


def credit_text(sidecar, include_routes=None, mask_routes=(), include_background=False):
    """Plain prompt data for scorers that do not use native tool-call messages."""
    lines = ['Question: ' + str(sidecar.get('question', ''))]
    for message in credit_messages(sidecar, include_routes, mask_routes, include_background):
        if message['role'] == 'assistant':
            lines.append('Tool call: ' + message['content'])
        else:
            lines.append('Tool result (' + message.get('name', 'tool') + '):\n' + message['content'])
    return '\n\n'.join(lines)
