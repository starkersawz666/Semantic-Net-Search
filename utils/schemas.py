# Generate statistics options from configuration
def generate_statistics(config):
    statistics = config['statistics']
    statistics_options = []
    for statistic in statistics:
        for key, value in statistic.items():
            statistics_options.append(key)
    return statistics_options

def generate_advanced_statistics(config):
    statistics = config['advanced_statistics']
    statistics_options = []
    for statistic in statistics:
        for key, value in statistic.items():
            statistics_options.append(key)
    return statistics_options

# Generate filter schema from configuration
def generate_schema(config):
    schema = []
    for item in config['basic_filters']:
        schema.append((item, 'bool'))
    for item in config['statistics']:
        for key, value in item.items():
            schema.append((key, value))
    return schema