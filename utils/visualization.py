import matplotlib.pyplot as plt

def plot_loss(history, save_path: str = None):
    """Plot training (and validation) loss vs. epochs using the history returned by model.fit()."""
    plt.figure()
    plt.plot(history.history.get('loss', []), label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
