import os
import matplotlib.pyplot as plt


def plot_model_losses(train_file, test_file, out_path, fig_show=True):
    # Read the train losses into a list of floats
    with open(train_file, 'r') as f:
        train_losses = [float(line.strip()) for line in f if line.strip()]
        
    # Read the test losses into a list of floats
    with open(test_file, 'r') as f:
        test_losses = [float(line.strip()) for line in f if line.strip()]
        
    # Initialize the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the data
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(test_losses, label='Test Loss', color='red', linewidth=2, linestyle='--')
    
    # Customize the plot with labels, title, and grid
    plt.title('FM Model Loss Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch / Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Display the plot
    plt.tight_layout()
    if fig_show:
        plt.show()
    else:
        plt.savefig(out_path)
        print(f"Succcessfully saved losses png at {out_path}")


if __name__ == "__main__":
    test_name = "test02"

    viz_path = f"../../../results/FM/{test_name}/losses/"
    out_path = f"../../../results/FM/{test_name}/viz/losses.png"
    
    plot_model_losses(os.path.join(viz_path, f"{test_name}_loss_train.txt"), 
                      os.path.join(viz_path, f"{test_name}_loss_test.txt"),
                      out_path, fig_show=False)
    