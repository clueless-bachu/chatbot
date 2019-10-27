
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config


def train(data, config, saved_model_dir):
	# loading nlu training samples
	training_data = load_data(data)

	# trainer to educate our pipeline
	trainer = Trainer(config.load(config))

	# train the model
	interpreter  = trainer.train(training_data)

	# store it for future use
	model_directory = trainer.persist(saved_model_dir,fixed_model_name="nlu")


if __name__ == '__main__':
	train("nlu.md","nlu_config.yml","./models/current")