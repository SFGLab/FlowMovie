import yaml
import os
import argparse


def prepare_CM_configs(test_name, template_path, CM_input_directory, CM_output_directory):
    # Create config directory if it doesn't exist
    config_out_path = os.path.join("/".join(template_path.split("/")[:-1]), test_name)
    if not os.path.exists(config_out_path):
        os.makedirs(config_out_path)

    # Load the base template
    with open(template_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Iterate through the target directories
    for phase_transition_folder in os.listdir(CM_input_directory):
        for trajectory_folder in os.listdir(os.path.join(CM_input_directory, phase_transition_folder)):

            # Update the specific parameters
            config_data['general']['input'] = os.path.join(CM_input_directory, phase_transition_folder, trajectory_folder)
            config_data['general']['output'] = os.path.join(CM_output_directory, phase_transition_folder, trajectory_folder)

            # Save the new YAML file
            save_path = os.path.join(config_out_path, phase_transition_folder+'_'+trajectory_folder+'.yaml')
            with open(save_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    print(f"Successfully generated configs in {config_out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ChromMovie configuration files for a test run."
    )

    # Default test_name used to construct dynamic default paths
    default_test = "test01"

    parser.add_argument(
        "-t",
        "--test-name",
        type=str,
        default=default_test,
        help=f"Name of the test run (default: {default_test})",
    )
    parser.add_argument(
        "--template-path",
        type=str,
        default="../../configs/md/base_CM_config.yaml",
        help="Path to base YAML template",
    )
    parser.add_argument(
        "-i",
        "--input-directory",
        type=str,
        default=f"/home/baneckik/Documents/projects/FlowMovie/data/md/{default_test}/CM_input/",
        help="Path to input directory",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        default=f"/home/baneckik/Documents/projects/FlowMovie/data/md/{default_test}/CM_output/",
        help="Path to output directory",
    )

    args = parser.parse_args()

    # Automatically update input/output paths if a custom test_name is passed
    # without explicitly providing custom input/output flags.
    if args.test_name != default_test:
        if args.input_directory.endswith(f"/{default_test}/CM_input/"):
            args.input_directory = f"/home/baneckik/Documents/projects/FlowMovie/data/md/{args.test_name}/CM_input/"
        if args.output_directory.endswith(f"/{default_test}/CM_output/"):
            args.output_directory = f"/home/baneckik/Documents/projects/FlowMovie/data/md/{args.test_name}/CM_output/"

    prepare_CM_configs(
        args.test_name,
        args.template_path,
        args.input_directory,
        args.output_directory,
    )


if __name__ == "__main__":
    main()