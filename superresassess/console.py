import typer
from pathlib import Path

from superresassess.data import HCP_T2W
from superresassess.preprocessing import PrepareHRLRData
from superresassess.experiments import setup_experiments

app = typer.Typer()


@app.command(
    help="This command will automatically download Human Connectome Data (HCP). It is"
    " necessary to have AWS CLI installed and the credentials setup to be able to"
    " download the data."
    "\n\nFor more information on how to do get the credentials, see this link:"
    " https://wiki.humanconnectome.org/docs/How%20To%20Connect%20to%20Connectome%20Data%20via%20AWS.html"
    "\n\nFor more information on how to install the AWS CLI, see this link:"
    " https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    "\n\nFor more information on how to setup AWS CLI with credentials, see this link:"
    " https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html"
)
def download(
    image_file: Path = typer.Argument(
        help="A text (.txt) file containing the paths to the images at the s3 bucket"
    ),
    image_folder: Path = typer.Argument(
        help="Path to folder were images should be downloaded."
    ),
):
    with open(image_file, "r") as fh:
        image_list = fh.readlines()

    image_list = [img.strip() for img in image_list]
    HCP_T2W(image_folder, image_list, download=True)


@app.command()
def preprocess(
    subject_folder_file: Path = typer.Argument(help="Path to folder with raw images."),
):
    # Get subject folders
    with open(subject_folder_file, "r") as fh:
        subject_folder_list = fh.readlines()

    # Strip newlines
    subject_folder_list = [subj.strip() for subj in subject_folder_list]

    # settings as defined in protocol
    prepare = PrepareHRLRData(
        lower_percentile=0.1,
        upper_percentile=99.9,
        lower_range_output=0,
        upper_range_output=255,
        smoothing_sigma=1,
        downsampling_scale=2,
    )
    for subject_folder in subject_folder_list:
        subject_folder = Path(subject_folder)
        processed_image_folder = subject_folder.joinpath("processed")
        prepare.prepare_data(".nii.gz", subject_folder, processed_image_folder)


@app.command()
def setup(
    study_configuration_file: Path = typer.Argument(
        help="A yaml file with the study configuration. For what information should be"
        " in the configuration file, please refer to"
        " superresassess.experiments.StudyConfiguration"
    ),
    experiment_folder: Path = typer.Argument(
        help="Path to folder where experiment configurations should be stored."
        " These experiment configurations will be according to"
        " superresassess.experiments.ExperimentConfiguraiton"
    ),
):
    setup_experiments(study_configuration_file, experiment_folder)


@app.command()
def run(
    image_label_file: Path = typer.Argument(
        help="Path to .txt file with on every line: a path to the lr file and the hr"
        " file respectively. The paths should be separated by a space and there should"
        " be no spaces in the path. Mind that this is not checked, this is up to the"
        " user to setup correctly"
    ),
    individual_experiment_folder: Path = typer.Argument(
        help="Path to folder with individual experiments."
    ),
    model_configuration_file: Path = typer.Argument(
        help="Path to a yaml file containing the model configuration."
        "\n\nFor details on the configuration see"
        " superresassess.model.ReCNNConfiguration"
    ),
    assessment_configuration_file: Path = typer.Argument(
        help="Path to a yaml file containing the assessment configuration."
        "\n\nFor details on the configuration see"
        " superresassess.assesment_methods.AssessmentConfig"
    ),
    log_dir: Path = typer.Option(
        default=Path("./logs"), help="Path to the logging folder"
    ),
): ...


def main():
    app()


if __name__ == "__main__":
    main()
