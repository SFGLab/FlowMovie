import yaml
import os


if __name__ == "__main__":

    test_name = 'test01'
    template_path = "./ChromMovie-flow/configs/md/base_CM_config.yaml"
    config_out_path = os.path.join("/".join(template_path.split("/")[:-1]), test_name)
    input_directory = f"./ChromMovie-flow/outputs/md/{test_name}"
    output_directory = f"./ChromMovie-flow/outputs/trajectories/{test_name}"
    target_extension = '.csv'

    # Create output directory if it doesn't exist
    if not os.path.exists(config_out_path):
        os.makedirs(config_out_path)

    # Load the base template
    with open(template_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Iterate through the target directories
    for phase_transition_folder in os.listdir(input_directory):
        for trajectory_folder in os.listdir(os.path.join(input_directory, phase_transition_folder)):
            
            # Update the specific parameters
            config_data['general']['input'] = os.path.join(input_directory, phase_transition_folder, trajectory_folder)
            config_data['general']['output'] = os.path.join(output_directory, phase_transition_folder, trajectory_folder)

            # Save the new YAML file
            save_path = os.path.join(config_out_path, phase_transition_folder+'_'+trajectory_folder+'.yaml')
            with open(save_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    print(f"Successfully generated configs in {output_directory}")