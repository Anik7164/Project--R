"""
Split tracked players into the two teams, plus goalkeepers and referees.

1. Collect shirt/shorts colour descriptors from clean (non-overlapping)
   player boxes.
2. Cluster them into two teams with k-means, ignoring outliers: the two
   teams are the two big groups, goalkeepers and referees wear colours
   that fit neither.
3. Label every box, then let each stable player ID vote over all of its
   boxes, so a few bad crops do not flip a player's team.
4. Outliers are split by position: a goalkeeper is nearly always the
   last person at one end of the play (and joins the team whose players
   are nearer); a referee stays in the middle of it.

Usage (after track_players.py):
    python src/identify_teams.py --video data/videos/test1.mp4
"""

from collections import Counter, defaultdict

import cv2
import numpy as np

from appearance import descriptor_distance, shirt_colour_name

TEAM_LABELS = ("team_a", "team_b")

# Minimum descriptors before fitting the team colours.
MIN_SAMPLES = 60
# Points farther than this many robust spreads from both team centres
# are outliers (goalkeeper, referee, or a bad crop).
OUTLIER_SPREAD = 3.0
# A player whose boxes were outliers at least this often is "other".
OTHER_VOTE_SHARE = 0.5
# An outlier counts as a goalkeeper if they are the leftmost or rightmost
# person in at least this share of the frames they appear in with
# others; a referee follows play from the middle.
GOALKEEPER_EDGE_SHARE = 0.7
# Goalkeeper IDs closer than this in colour wear the same kit.
SAME_KIT_DISTANCE = 0.3


def _embed(descriptors):
    # sqrt turns the Hellinger distance into a Euclidean one.
    return np.sqrt(np.asarray(descriptors, dtype=np.float32))


def _kmeans(points, k, attempts=5):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    _, labels, centres = cv2.kmeans(
        points, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    return labels.ravel(), centres


class TeamClassifier:
    """Two team colour models fitted from player descriptors."""

    def __init__(self):
        self.centres = None
        self.spread = None
        self.colours = {}

    @property
    def fitted(self):
        return self.centres is not None

    def fit(self, descriptors):
        descriptors = [d for d in descriptors if d is not None]
        if len(descriptors) < MIN_SAMPLES:
            raise ValueError(
                f"Need at least {MIN_SAMPLES} player crops to find the teams, "
                f"got {len(descriptors)}."
            )
        points = _embed(descriptors)

        # Fit, drop far points, refit: goalkeepers and the referee are few
        # and would otherwise drag a team centre towards their colour.
        keep = np.ones(len(points), dtype=bool)
        for _ in range(3):
            labels, centres = _kmeans(points[keep], 2)
            distances = np.linalg.norm(
                points[:, None, :] - centres[None, :, :], axis=2
            )
            nearest = distances.min(axis=1)
            spread = np.median(nearest[keep]) + 1e-6
            keep = nearest < OUTLIER_SPREAD * spread

        self.centres = centres
        self.spread = spread
        nearest_team = distances.argmin(axis=1)
        for team_index, label in enumerate(TEAM_LABELS):
            members = [
                d for d, t, k in zip(descriptors, nearest_team, keep)
                if t == team_index and k
            ]
            mean = np.mean(members, axis=0) if members else None
            self.colours[label] = shirt_colour_name(mean)
        return self

    def predict(self, descriptor):
        """Return (label, distance) for one descriptor.

        label is "team_a", "team_b", "other" (fits neither colour) or
        None when there is no descriptor.
        """
        if descriptor is None or not self.fitted:
            return None, None
        point = _embed([descriptor])[0]
        distances = np.linalg.norm(self.centres - point, axis=1)
        index = int(distances.argmin())
        if distances[index] >= OUTLIER_SPREAD * self.spread:
            return "other", float(distances[index])
        return TEAM_LABELS[index], float(distances[index])


class TeamVotes:
    """Accumulates per-box labels and decides each player's team and role."""

    def __init__(self):
        self.votes = defaultdict(Counter)
        # For outliers: whether they were the leftmost/rightmost person in
        # each frame, and which team is nearer horizontally.
        self.at_edge = defaultdict(list)
        self.nearer_team = defaultdict(Counter)
        self.outlier_colour = {}

    def add_frame(self, player_ids, boxes, labels, descriptors=None):
        """Record one frame: stable player IDs, their boxes, box labels and
        (optionally) descriptors, used to tell the two goalkeepers apart."""
        centres_x = [(b[0] + b[2]) / 2.0 for b in boxes]
        order = np.argsort(np.argsort(centres_x))
        last = len(boxes) - 1

        team_x = defaultdict(list)
        for x, label in zip(centres_x, labels):
            if label in TEAM_LABELS:
                team_x[label].append(x)
        team_mean = {t: float(np.mean(xs)) for t, xs in team_x.items()}

        for index, (player_id, label) in enumerate(zip(player_ids, labels)):
            if label is None:
                continue
            self.votes[player_id][label] += 1
            if label != "other":
                continue
            if descriptors is not None and descriptors[index] is not None:
                old = self.outlier_colour.get(player_id)
                self.outlier_colour[player_id] = (
                    descriptors[index] if old is None
                    else 0.9 * old + 0.1 * descriptors[index]
                )
            if len(boxes) >= 3:
                self.at_edge[player_id].append(order[index] in (0, last))
            if len(team_mean) == 2:
                nearer = min(
                    team_mean,
                    key=lambda t: abs(team_mean[t] - centres_x[index]),
                )
                self.nearer_team[player_id][nearer] += 1

    def decide(self):
        """Return {player_id: {"team", "role", "confidence"}}.

        team is "team_a", "team_b" or None (referee / unknown); role is
        "player", "goalkeeper" or "referee".
        """
        decisions = {}
        for player_id, votes in self.votes.items():
            total = sum(votes.values())
            other_share = votes["other"] / total
            team_votes = {t: votes[t] for t in TEAM_LABELS}

            if other_share < OTHER_VOTE_SHARE:
                team = max(team_votes, key=team_votes.get)
                decisions[player_id] = {
                    "team": team,
                    "role": "player",
                    "confidence": round(team_votes[team] / total, 3),
                }
                continue

            at_edge = self.at_edge[player_id]
            edge_share = float(np.mean(at_edge)) if at_edge else 0.0
            if edge_share >= GOALKEEPER_EDGE_SHARE and self.nearer_team[player_id]:
                team = self.nearer_team[player_id].most_common(1)[0][0]
                decisions[player_id] = {
                    "team": team,
                    "role": "goalkeeper",
                    "confidence": round(other_share, 3),
                }
            else:
                decisions[player_id] = {
                    "team": None,
                    "role": "referee",
                    "confidence": round(other_share, 3),
                }
        self._separate_goalkeepers(decisions)
        return decisions

    def _separate_goalkeepers(self, decisions):
        """The two goalkeeper kits belong to different teams.

        One goalkeeper can have several IDs, so group them by kit colour.
        If both kits ended up in one team, move the kit with the weaker
        team vote to the other team.
        """
        kits = []  # [(colour, [player_ids])]
        for player_id, decision in decisions.items():
            colour = self.outlier_colour.get(player_id)
            if decision["role"] != "goalkeeper" or colour is None:
                continue
            for kit_colour, members in kits:
                if descriptor_distance(kit_colour, colour) < SAME_KIT_DISTANCE:
                    members.append(player_id)
                    break
            else:
                kits.append((colour, [player_id]))
        if len(kits) != 2:
            return

        def team_votes(members):
            votes = Counter()
            for player_id in members:
                votes.update(self.nearer_team[player_id])
            return votes

        votes = [team_votes(members) for _, members in kits]
        teams = [v.most_common(1)[0][0] for v in votes]
        if teams[0] != teams[1]:
            return
        margins = [v[teams[0]] / max(1, sum(v.values())) for v in votes]
        weaker = int(np.argmin(margins))
        other_team = next(t for t in TEAM_LABELS if t != teams[0])
        for player_id in kits[weaker][1]:
            decisions[player_id]["team"] = other_team
