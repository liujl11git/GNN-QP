# tensorflow=2.16

import argparse
import logging
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

from models import GCNPolicy
from pathlib import Path
from tqdm import tqdm
from utils import load_data_folder, prep_batch_data


MODEL_DICT = {
    'GNN': GCNPolicy,
}

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0,
                    help="index of the gpu used for training")
parser.add_argument("--emb-size", type=int, default=6,
                    help="embedding size of hidden states in 2-FGNN")
parser.add_argument("--num-layers", type=int, default=2,
                    help="number of graph convolution layers in the network")
parser.add_argument("--task", type=str, choices=["objective", "solution"],
                    help="task to conduct in this run of experiment")
parser.add_argument("--lr", type=float, default=3e-4,
                    help="initial learning rate")
parser.add_argument("--weight-decay", type=float, default=None,
                    help="weight decay")
parser.add_argument("--lr-decay", type=str, default=None,
                    help="learning rate scheduling")
parser.add_argument("--num-epochs", type=int, default="10000",
                    help="num of epochs for training")
parser.add_argument("--data-path", type=str,
                    help="path to the directory that contains training data")
parser.add_argument("--valid-data-path", type=str, default=None,
                    help="path to the directory that contains validation data")
parser.add_argument("--model", type=str, choices=['GNN'],
                    help="type of model that is trained")
parser.add_argument("--gather-first", action="store_true",
                    help="enable to use the mixed-integer extended version")
parser.add_argument("--optimizer", type=str, default='Adam', choices=['Adam', 'SGD'],
                    help="type of optimizer to use")
parser.add_argument("--batch-size", type=int, default=1,
                    help="batch size for training and evaluating")
parser.add_argument("--seed", type=int, default=1812,
                    help="random seed for reproducibility")
parser.add_argument("--save-path", type=str, default='./results/default',
                    help="path where checkpoints and logs are saved")
parser.add_argument("--best-wait", type=int, default=-1,
                    help="number of epochs of waiting for better training loss")
parser.add_argument("--full-gd", action="store_true",
                    help="full gradient descent on the whole training set")


def setup_logger(logdir):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logdir = Path(logdir)
    logging.basicConfig(
        format="[%(asctime)s] [%(name)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(logdir/'log.txt', mode='w'),
                  logging.StreamHandler(os.sys.stdout)]
    )
    return logging.getLogger('main')


def process(model, dataset, optimizer, args, valid_dataset=None):
    """Train the network for one epoch.
    """
    (cons_features, edge_indices, edge_features, var_features, Q_matrices, labels) = dataset

    num_samples = len(cons_features)
    order = np.arange(num_samples, dtype=int)
    np.random.shuffle(order)
    batch_size = args.batch_size

    train_vars = model.variables
    accumulated_loss = 0.0
    accumulated_rel_error = 0.0
    batch_count = 0
    if args.full_gd:
        accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]

    for start in tqdm(range(0, num_samples, batch_size)):
        with tf.GradientTape() as tape:
            if batch_size == 1:
                selected = order[start]
                inputs = (cons_features[selected], edge_indices[selected],
                          edge_features[selected], var_features[selected],
                          Q_matrices[selected],)
                label = labels[selected]
                cons_segments = var_segments = var_repeats = None
            else:
                end = min(start + batch_size, num_samples)
                selected = order[start:end]
                inputs, label, cons_segments, var_segments, var_repeats = prep_batch_data(
                    [cons_features[idx] for idx in selected],
                    [edge_indices[idx] for idx in selected],
                    [edge_features[idx] for idx in selected],
                    [var_features[idx] for idx in selected],
                    [Q_matrices[idx] for idx in selected],
                    [labels[idx] for idx in selected],
                )

            out = model(inputs,
                        training=True,
                        cons_segments=cons_segments,
                        var_segments=var_segments,
                        var_repeats=var_repeats)
            label = tf.reshape(label, out.shape)

            if args.task == "objective":
                error = (label - out)**2.0
                denom = tf.math.maximum(label**2, 1.0)
                loss = tf.reduce_mean(error / denom)
                rel_error = tf.math.sqrt(loss)
            elif args.task == "solution":
                if batch_size == 1:
                    denom = tf.reduce_sum(label**2.0)
                    denom = tf.math.maximum(denom, 1.0)
                    error = tf.reduce_sum((label - out) ** 2.0)
                else:
                    denom = tf.math.segment_sum(label**2.0, var_segments)
                    denom = tf.math.maximum(denom, 1.0)
                    error = tf.math.segment_sum((label - out) ** 2.0, var_segments)
                loss = tf.reduce_mean(error / denom)
                rel_error = tf.reduce_mean(tf.sqrt(error / denom))
            else:
                raise NotImplementedError(f"Loss function for task {args.task} is not implemented yet.")

            grads = tape.gradient(target=loss, sources=train_vars)
            accumulated_loss += loss.numpy()
            accumulated_rel_error += rel_error.numpy()
            if args.full_gd:
                accum_gradient = [(accum_grad + grad)
                                  for accum_grad,grad in zip(accum_gradient, grads)]
            else:
                optimizer.apply_gradients(zip(grads, train_vars))

            batch_count += 1

    if args.full_gd:
        accum_gradient = [this_grad / batch_count for this_grad in accum_gradient]
        optimizer.apply_gradients(zip(accum_gradient, train_vars))

    if valid_dataset is not None:
        valid_loss, valid_rel_error = process_valid(model, valid_dataset, args)
    else:
        valid_loss, valid_rel_error = None, None

    ret = (accumulated_loss / batch_count,
           accumulated_rel_error / batch_count,
           valid_loss,
           valid_rel_error)

    return ret


def process_valid(model, dataset, args):
    """Evaluate the network for one epoch on the test set.
    """
    (cons_features, edge_indices, edge_features, var_features, Q_matrices, labels) = dataset

    num_samples = len(cons_features)
    batch_size = args.batch_size

    losses = []
    rel_errors = []
    for start in tqdm(range(0, num_samples, batch_size)):
        if batch_size == 1:
            inputs = (cons_features[start], edge_indices[start],
                        edge_features[start], var_features[start],
                        Q_matrices[start],)
            label = labels[start]
            cons_segments = var_segments = var_repeats = None
        else:
            end = min(start + batch_size, num_samples)
            inputs, label, cons_segments, var_segments, var_repeats = prep_batch_data(
                cons_features[start:end],
                edge_indices[start:end],
                edge_features[start:end],
                var_features[start:end],
                Q_matrices[start:end],
                labels[start:end]
            )

        out = model(inputs,
                    training=False,
                    cons_segments=cons_segments,
                    var_segments=var_segments,
                    var_repeats=var_repeats)
        label = tf.reshape(label, out.shape)

        if args.task == "objective":
            error = (label - out)**2.0
            denom = tf.math.maximum(label**2, 1.0)
            loss = tf.reduce_mean(error / denom)
            rel_error = tf.math.sqrt(loss)
        elif args.task == "solution":
            if batch_size == 1:
                denom = tf.math.maximum(tf.norm(label, ord=2, axis=1), 1.0)
                error = tf.reduce_sum((label - out) ** 2.0, axis=1)
            else:
                denom = tf.math.segment_sum(label**2.0, var_segments)
                denom = tf.math.maximum(denom, 1.0)
                error = tf.math.segment_sum((label - out) ** 2.0, var_segments)
            loss = tf.reduce_mean(error / denom)
            rel_error = tf.reduce_mean(tf.sqrt(error / denom))
        else:
            raise NotImplementedError(f"Loss function for task {args.task} is not implemented yet.")

        losses.append(loss.numpy())
        rel_errors.append(rel_error.numpy())

    return np.mean(losses), np.mean(rel_errors)


if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logger
    os.makedirs(args.save_path, exist_ok=True)
    logger = setup_logger(args.save_path)
    model_save_path = os.path.join(args.save_path, 'model.pkl')

    # Set up randomness (for reproducibility)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed+1)

    # Set up computation resources
    gpu_index = int(args.gpu)
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpu_index], True)

    with tf.device(f"GPU:{gpu_index}"):

        # Set up dataset
        (var_features, cons_features, edge_features, edge_indices,
         Q_matrices, labels_obj, labels_solu,
         var_dim, cons_dim, edge_dim) = load_data_folder(args.data_path)
        labels = labels_obj if args.task == "objective" else labels_solu
        train_data = (cons_features, edge_indices, edge_features, var_features,
                      Q_matrices, labels)

        if args.valid_data_path is not None:
            (valid_var_features, valid_cons_features,
             valid_edge_features, valid_edge_indices,
             valid_Q_matrices, valid_labels_obj, valid_labels_solu,
             _, _, _) = load_data_folder(args.valid_data_path)
            valid_labels = valid_labels_obj if args.task == "objective" else valid_labels_solu
            valid_data = (valid_cons_features, valid_edge_indices,
                          valid_edge_features, valid_var_features,
                          valid_Q_matrices, valid_labels)
        else:
            valid_data = None

		# Model and optimizer initialization
        model = MODEL_DICT[args.model](args.emb_size, args.num_layers,
                                       cons_dim, edge_dim, var_dim,
                                       gather_first=args.gather_first,
                                       isGraphLevel=args.task=="objective")

        cur_lr = args.lr
        if args.optimizer == 'Adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=cur_lr, weight_decay=args.weight_decay,
            )
        elif args.optimizer == 'SGD':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=cur_lr, weight_decay=args.weight_decay,
            )

        # Model checkpoint
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(ckpt, args.save_path, max_to_keep=1)

        loss_best = 1e20
        best_epoch = -1
        epoch = 0

        ### MAIN LOOP ###
        while epoch < args.num_epochs:
            train_loss, train_rel_error, valid_loss, valid_rel_error = process(
                model, train_data, optimizer, args, valid_dataset=valid_data
            )

            logger.info(f"Epoch: {epoch}\tTrain Loss: {train_loss:.8e}\tTrain Rel-Error: {train_rel_error}")
            if valid_data is not None:
                logger.info(f"              \tValid Loss: {valid_loss:.8e}\tValid Rel-Error: {valid_rel_error}")

            if train_loss < loss_best:
                loss_best = train_loss
                best_epoch = epoch
                model_save_path = manager.save()
                logger.info(f"Saved checkpoint with better training loss: {model_save_path}")

            elif cur_lr >= 1e-10 and args.best_wait > 0 and epoch - best_epoch >= args.best_wait:
                logger.info(f"Loss has not improved for {args.best_wait} epochs.")
                ckpt.restore(manager.latest_checkpoint)
                logger.info(f"Restored last checkpoint at epoch {best_epoch} with best training loss.")

                cur_lr *= 0.5
                logger.info(f"Learning rate decayed to {cur_lr}")
                optimizer.learning_rate.assign(cur_lr)

                epoch = best_epoch

            epoch += 1
            ckpt.step.assign(epoch)

        model.summary()
