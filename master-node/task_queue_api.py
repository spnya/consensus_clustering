from flask import Blueprint, request, jsonify
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import logging
from contextlib import contextmanager
import itertools
from sklearn.datasets import make_blobs, make_moons, make_circles

import numpy as np
from sklearn.preprocessing import StandardScaler

from models import Task, Result, Worker, Experiment, get_db_session
from task_queue_service import TaskQueueService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _positive_int(value, default):
    """Return *value* coerced to int if it is >= 1, otherwise *default*."""
    try:
        val = int(value)
    except (TypeError, ValueError):
        return default
    return val if val > 0 else default


def frange(start: float, stop: float, step: float):
    """Floating‑point range generator inclusive of *stop*."""
    while start <= stop + 1e-9:  # numeric guard against FP error
        yield round(start, 12)
        start += step


# ---------------------------------------------------------------------------
# Flask blueprint and task queue setup
# ---------------------------------------------------------------------------

task_queue = TaskQueueService()

task_api = Blueprint("task_api", __name__)


# Context manager for database sessions
@contextmanager
def session_scope():
    session = get_db_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def generate_custom_synthetic_data(config):
    n_samples = config.get("n_samples", 300)
    n_features = config.get("n_features", 2)
    distribution = config.get("distribution", "uniform").lower()
    random_state = config.get("random_state", None)
    rng = np.random.default_rng(random_state)

    if distribution == "uniform":
        low = config.get("low", -1)
        high = config.get("high", 1)
        X = rng.uniform(low, high, size=(n_samples, n_features))

    elif distribution == "zipf":
        a = config.get("a", 2.0)
        raw = rng.zipf(a, size=(n_samples, n_features))
        X = raw.astype(float)
        X = StandardScaler().fit_transform(X)

    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    y = np.zeros(n_samples, dtype=int)  # Dummy labels
    return X, y


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@task_api.route("/experiments/<int:experiment_id>/generate", methods=["POST"])
def generate_experiment_tasks(experiment_id):
    """Generate tasks for an experiment and enqueue them.

    Returns JSON with the number of tasks queued or an error message.
    """
    logger.info("🔥 HIT generate_experiment_tasks for experiment %s", experiment_id)

    try:
        with session_scope() as session:
            experiment = (
                session.query(Experiment).filter_by(id=experiment_id).first()
            )
            if not experiment:
                return jsonify({"error": "Experiment not found"}), 404

            # ----------------------------------------------------------------
            # Parameter extraction + validation
            # ----------------------------------------------------------------
            params = experiment.parameters or {}

            regular_algos = (
                params.get("base_algorithms")
                or params.get("regular_algorithms", [])
            )
            consensus_algos = params.get("consensus_algorithms", [])

            # Fail fast if no algorithms provided
            if not regular_algos and not consensus_algos:
                return (
                    jsonify({"error": "No algorithms specified in experiment parameters"}),
                    400,
                )

            ensemble_size = _positive_int(params.get("ensemble_size", 5), 5)
            mutation_step = float(params.get("mutation_step", 0.1))
            if mutation_step <= 0:
                mutation_step = 0.1
            mutation_runs = _positive_int(params.get("runs_per_step", 3), 3)
            dataset_runs = _positive_int(params.get("dataset_runs", 1), 1)
            cluster_config = params.get("synthetic_config", {})
            tuning_grid = params.get("tuning_grid", {})

            # Derived settings
            shift_types = ["none", "scale", "modularity"]
            # Cap mutation grid to avoid explosion (max 200 points)
            mutation_probs = list(
                itertools.islice((round(p, 3) for p in frange(0.0, 1.0, mutation_step)), 200)
            )
            synthetic_generator = cluster_config.get("generator", "blobs")

            logger.debug(
                "Generating tasks: datasets=%d, ensemble=%d, runs/step=%d, "
                "regular=%s, consensus=%s, mutation_points=%d",
                dataset_runs,
                ensemble_size,
                mutation_runs,
                regular_algos,
                consensus_algos,
                len(mutation_probs),
            )

            # ----------------------------------------------------------------
            # Task generation loops
            # ----------------------------------------------------------------
            generated_tasks = []

            for dataset_index in range(dataset_runs):
                # 1. Create synthetic data set
                if synthetic_generator == "moons":
                    X, y_true = make_moons(
                        n_samples=cluster_config.get("n_samples", 300),
                        noise=cluster_config.get("noise", 0.1),
                    )
                elif synthetic_generator == "circles":
                    X, y_true = make_circles(
                        n_samples=cluster_config.get("n_samples", 300),
                        noise=cluster_config.get("noise", 0.05),
                        factor=cluster_config.get("factor", 0.5),
                    )
                elif synthetic_generator == "custom":
                    X, y_true = generate_custom_synthetic_data(cluster_config)
                else:  # default to blobs
                    X, y_true = make_blobs(
                        n_samples=cluster_config.get("n_samples", 300),
                        centers=cluster_config.get("n_clusters", 4),
                        cluster_std=cluster_config.get("cluster_std", 1.5),
                    )

                # 2. Regular/base algorithm tasks
                for ensemble_member in range(ensemble_size):
                    for algo in regular_algos:
                        grid = tuning_grid.get(algo, {})
                        param_keys = list(grid.keys())
                        param_values = [grid[k] for k in param_keys]
                        grid_combinations = (
                            list(itertools.product(*param_values)) if param_values else [()]
                        )

                        for params_tuple in grid_combinations:
                            algo_params = dict(zip(param_keys, params_tuple))
                            for shift in shift_types:
                                for mp in mutation_probs:
                                    for run_id in range(mutation_runs):
                                        task_data = {
                                            "operation": algo,
                                            "algorithm": algo,
                                            "shift_type": shift,
                                            "mutation_prob": mp,
                                            "synthetic_config": cluster_config,
                                            "params": algo_params,
                                            "data": X.tolist(),
                                            "ground_truth": y_true.tolist(),
                                            "dataset_index": dataset_index,
                                            "ensemble_member": ensemble_member,
                                            "run_type": "base",
                                        }

                                        new_task = Task(
                                            experiment_id=experiment_id,
                                            task_data=task_data,
                                            status="pending",
                                        )
                                        session.add(new_task)
                                        session.flush()  # obtain id for to_dict()
                                        generated_tasks.append(new_task.to_dict())

            # ----------------------------------------------------------------
            # Queue the tasks for workers
            # ----------------------------------------------------------------
            for task_dict in generated_tasks:
                task_queue.enqueue_task(task_dict)

        # --------------------------------------------------------------------
        # Response
        # --------------------------------------------------------------------
        return (
            jsonify(
                {
                    "generated": len(generated_tasks),
                    "sample_task": generated_tasks[0] if generated_tasks else None,
                }
            ),
            201,
        )

    except SQLAlchemyError as db_exc:
        logger.error("Database error during experiment generation: %s", db_exc)
        return jsonify({"error": "Database error"}), 500


@task_api.route("/tasks/request", methods=["GET"])
def request_task():
    worker_id = request.args.get("worker_id")

    if not worker_id:
        return jsonify({"error": "worker_id is required"}), 400

    try:
        with session_scope() as session:
            worker = session.query(Worker).filter_by(id=worker_id).first()
            if not worker:
                return jsonify({"error": "Worker not found"}), 404

            # heartbeat
            worker.last_heartbeat = datetime.utcnow()
            worker.status = "active"

            metadata = worker.worker_metadata or {}
            max_tasks = metadata.get("max_tasks", 2)
            active_tasks = metadata.get("active_tasks", 0)

            task_queue.update_worker_stats(
                worker_id,
                {
                    "max_tasks": max_tasks,
                    "active_tasks": active_tasks,
                },
            )

    except SQLAlchemyError as db_exc:
        logger.error("Database error: %s", db_exc)
        return jsonify({"error": "Database error"}), 500

    # Try to assign a task (or steal one if idle)
    task = task_queue.assign_task(worker_id) or task_queue.steal_work(worker_id)

    if not task:
        return "", 204

    return jsonify(task), 200


@task_api.route("/tasks/submit", methods=["POST"])
def submit_task_result():
    data = request.json or {}

    required = {"task_id", "worker_id", "result_data"}
    if not required.issubset(data):
        return (
            jsonify({"error": "task_id, worker_id, and result_data are required"}),
            400,
        )

    task_id = data["task_id"]
    worker_id = data["worker_id"]
    result_data = data["result_data"]
    processing_time = data.get("processing_time")

    if not task_queue.complete_task(task_id, worker_id, result_data):
        return jsonify({"error": "Task not assigned to this worker"}), 403

    try:
        with session_scope() as session:
            task = session.query(Task).filter_by(id=task_id).first()
            if not task:
                return jsonify({"error": "Task not found"}), 404

            task.status = "completed"
            task.updated_at = datetime.utcnow()

            new_result = Result(
                task_id=task_id,
                worker_id=worker_id,
                result_data=result_data,
                processing_time=processing_time,
            )
            session.add(new_result)
            session.flush()

            return jsonify(new_result.to_dict()), 201

    except SQLAlchemyError as db_exc:
        logger.error("Database error: %s", db_exc)
        return jsonify({"error": "Database error"}), 500


@task_api.route("/queue/stats", methods=["GET"])
def get_queue_stats():
    return jsonify(task_queue.get_queue_stats()), 200


# ---------------------------------------------------------------------------
# Blueprint registration helper
# ---------------------------------------------------------------------------

def init_app(app):
    """Call from your application factory to register endpoints."""
    app.register_blueprint(task_api, url_prefix="/api")
