from rasa_core.policies import FallbackPolicy, KerasPolicy, MemoizationPolicy
from rasa_core.agent import Agent


def train_core(domain,data ,saved_model_dir):
	# this will catch predictions the model isn't very certain about
	# there is a threshold for the NLU predictions as well as the action predictions
	fallback = FallbackPolicy(fallback_action_name="action_default_fallback",core_threshold=0.2,nlu_threshold=0.2)
	agent = Agent(domain, policies=[MemoizationPolicy(), KerasPolicy(), fallback])
	training_data = agent.load_data(data)
	agent.train(
	    training_data,
	    validation_split=0.0
	)
	agent.persist(saved_model_dir)


if __name__ == '__main__':
	train_core('domain.yml','stories.md','models/dialogue')
    