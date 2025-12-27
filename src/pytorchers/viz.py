import torch
from matplotlib import pyplot as plt
from torch import nn


class ForwardTracker:
    def __init_subclass__(cls):
        if not issubclass(cls, nn.Module):
            raise TypeError(
                f"to subclass TrailRecorder, Class {cls.__name__} must subclass from nn.Module too"
            )

    def forward_track(self):
        if hasattr(self, "forward_tracking"):
            return
        else:
            self.forward_tracking = True

        def gen_activation_hook(name):
            def hook(module, input, output):
                if not hasattr(self, "activations"):
                    self.activations = {}
                self.activations[f"{name}_activations"] = output

            return hook

        for name, child in self.named_children():
            if isinstance(child, nn.Linear):
                child.register_forward_hook(gen_activation_hook(name))

    def plot_activations(self, X, y, agg_func=torch.mean, fig_title=""):
        self.forward_track()
        preds = self(X)
        num_layers = len(self.activations)

        fig, axs = plt.subplots(1, num_layers, figsize=(5 * (num_layers + 1), 8))

        for i, (name, activation) in enumerate(list(self.activations.items())[:-1]):
            agg_activation = agg_func(activation, dim=0).detach().numpy()
            axs[i].imshow(agg_activation.reshape(-1, 1), cmap="YlGn", aspect="auto")
            axs[i].set(
                title=f"{name}",
                xlabel="",
                ylabel="",
                xticklabels=[],
                yticklabels=[],
            )

        # Handle both regression and classification
        y_numpy = y.detach().numpy()
        if preds.dim() > 1 and preds.shape[1] > 1:
            # Classification: take argmax to get predicted classes
            preds_numpy = preds.argmax(dim=1).detach().numpy()
        else:
            # Regression: use predictions directly
            preds_numpy = preds.detach().numpy()

        axs[-1].scatter(y_numpy, preds_numpy, alpha=0.5, color="yellowgreen")
        axs[-1].plot(y_numpy, y_numpy, "r--")
        axs[-1].set(title="truth vs preds", xlabel="truth", ylabel="preds")

        fig.suptitle(fig_title)
        fig.tight_layout()
        fig.show()

    def plot_compared_activations(
        self, dataset1, dataset2, agg_func=torch.mean, fig_title1="", fig_title2=""
    ):
        self.plot_activations(*dataset1, agg_func, fig_title=fig_title1)
        self.plot_activations(*dataset2, agg_func, fig_title=fig_title2)
