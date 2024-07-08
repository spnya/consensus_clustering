import logging
import os

from flask import Flask, request, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin, BaseView, expose
from flask_admin.contrib.sqla import ModelView
from clustering import generate_ground_truth, mutate_partition, compute_consensus_matrix, modularity_shift, scale_shift
from clustering import agglomerative_clustering, louvain_clustering, birch_clustering, optics_clustering
from clustering import spectral_clustering, meanshift_clustering, kmeans_clustering
import numpy as np
from sklearn.metrics import adjusted_rand_score
import threading

app = Flask(__name__)
lock = threading.Lock()
semaphore = threading.Semaphore(5)

# Configure the database
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{os.environ.get("DB_USER")}:{os.environ.get("DB_PASSWORD")}@82.97.244.247:5432/cluster'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)

app.app_context().push()

# Define the Experiment model
class Experiment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    num_objects = db.Column(db.Integer, nullable=False)
    num_clusters = db.Column(db.Integer, nullable=False)
    ensemble_size = db.Column(db.Integer, nullable=False)
    mutation_probability = db.Column(db.Float, nullable=False)
    shift_type = db.Column(db.String(50), nullable=False)
    agglomerative_clusters = db.Column(db.Integer, nullable=False)
    agglomerative_ari = db.Column(db.Float, nullable=False)
    louvain_clusters = db.Column(db.Integer, nullable=False)
    louvain_ari = db.Column(db.Float, nullable=False)
    birch_clusters = db.Column(db.Integer, nullable=False)
    birch_ari = db.Column(db.Float, nullable=False)
    optics_clusters = db.Column(db.Integer, nullable=False)
    optics_ari = db.Column(db.Float, nullable=False)
    spectral_clusters = db.Column(db.Integer, nullable=False)
    spectral_ari = db.Column(db.Float, nullable=False)
    meanshift_clusters = db.Column(db.Integer, nullable=False)
    meanshift_ari = db.Column(db.Float, nullable=False)
    kmeans_clusters = db.Column(db.Integer, nullable=False)
    kmeans_ari = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

# Initialize Flask-Admin
admin = Admin(app, name='Experiment Admin', template_mode='bootstrap3')
admin.add_view(ModelView(Experiment, db.session))

experiments = []
experiment_results = []
current_experiment_index = 0
lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/experiment')
def experiment():
    return render_template('experiment.html')

@app.route('/run_experiments', methods=['POST'])
def run_experiments():
    global experiments, experiment_results, current_experiment_index
    data = request.json
    N = data['N']
    K = data['K']
    M = data['M']
    p = data['p']
    shift_type = data['shift_type']
    num_experiments = data['num_experiments']
    use_power_law = data.get('use_power_law', False)
    alpha = data.get('alpha', 1.5)

    experiments = [{'N': N, 'K': K, 'M': M, 'p': p, 'shift_type': shift_type, 'use_power_law': use_power_law, 'alpha': alpha} for _ in range(num_experiments)]
    experiment_results = []
    current_experiment_index = 0

    def run_experiment_thread(exp):
        with app.app_context():
            with semaphore:
                result = run_single_experiment(exp)
                with lock:
                    experiment_results.append(result)
                    save_result_to_db(exp, result)
                    global current_experiment_index
                    current_experiment_index += 1

    def run_all_experiments():
        with app.app_context():
            global experiments
            threads = []
            for exp in experiments:
                thread = threading.Thread(target=run_experiment_thread, args=(exp,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

    threading.Thread(target=run_all_experiments).start()
    return jsonify({'status': 'Experiments started'})

@app.route('/status', methods=['GET'])
def status():
    global experiments, experiment_results, current_experiment_index
    with lock:
        completed = len(experiment_results)
        total = len(experiments)
        return jsonify({
            'completed': completed,
            'total': total,
            'results': experiment_results
        })

def run_single_experiment(exp):
    N = exp['N']
    K = exp['K']
    M = exp['M']
    p = exp['p']
    shift_type = exp['shift_type']
    use_power_law = exp['use_power_law']
    alpha = exp['alpha']

    ground_truth = generate_ground_truth(N, K, use_power_law, alpha)
    ensemble = [mutate_partition(ground_truth, p) for _ in range(M)]
    consensus_matrix = compute_consensus_matrix(ensemble)

    if shift_type == 'modularity':
        consensus_matrix = modularity_shift(consensus_matrix)
    else:
        consensus_matrix = scale_shift(consensus_matrix)

    agglomerative_result = agglomerative_clustering(consensus_matrix, K)
    louvain_result = louvain_clustering(consensus_matrix, K)
    birch_result = birch_clustering(consensus_matrix, K)
    optics_result = optics_clustering(consensus_matrix)
    spectral_result = spectral_clustering(consensus_matrix, K)
    meanshift_result = meanshift_clustering(consensus_matrix)
    kmeans_result = kmeans_clustering(consensus_matrix, K)

    agglomerative_ari = adjusted_rand_score(ground_truth, agglomerative_result)
    louvain_ari = adjusted_rand_score(ground_truth, louvain_result)
    birch_ari = adjusted_rand_score(ground_truth, birch_result)
    optics_ari = adjusted_rand_score(ground_truth, optics_result)
    spectral_ari = adjusted_rand_score(ground_truth, spectral_result)
    meanshift_ari = adjusted_rand_score(ground_truth, meanshift_result)
    kmeans_ari = adjusted_rand_score(ground_truth, kmeans_result)

    return {
        'agglomerative': {
            'clusters': len(np.unique(agglomerative_result)),
            'ari': agglomerative_ari
        },
        'louvain': {
            'clusters': len(np.unique(louvain_result)),
            'ari': louvain_ari
        },
        'birch': {
            'clusters': len(np.unique(birch_result)),
            'ari': birch_ari
        },
        'optics': {
            'clusters': len(np.unique(optics_result)),
            'ari': optics_ari
        },
        'spectral': {
            'clusters': len(np.unique(spectral_result)),
            'ari': spectral_ari
        },
        'meanshift': {
            'clusters': len(np.unique(meanshift_result)),
            'ari': meanshift_ari
        },
        'kmeans': {
            'clusters': len(np.unique(kmeans_result)),
            'ari': kmeans_ari
        }
    }

def save_result_to_db(exp, result):
    experiment = Experiment(
        num_objects=exp['N'],
        num_clusters=exp['K'],
        ensemble_size=exp['M'],
        mutation_probability=exp['p'],
        shift_type=exp['shift_type'],
        agglomerative_clusters=result['agglomerative']['clusters'],
        agglomerative_ari=result['agglomerative']['ari'],
        louvain_clusters=result['louvain']['clusters'],
        louvain_ari=result['louvain']['ari'],
        birch_clusters=result['birch']['clusters'],
        birch_ari=result['birch']['ari'],
        optics_clusters=result['optics']['clusters'],
        optics_ari=result['optics']['ari'],
        spectral_clusters=result['spectral']['clusters'],
        spectral_ari=result['spectral']['ari'],
        meanshift_clusters=result['meanshift']['clusters'],
        meanshift_ari=result['meanshift']['ari'],
        kmeans_clusters=result['kmeans']['clusters'],
        kmeans_ari=result['kmeans']['ari'],
    )
    db.session.add(experiment)
    db.session.commit()

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True, port=8000)
