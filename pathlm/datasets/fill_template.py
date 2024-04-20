from argparse import ArgumentParser
from random import choice
from pathlm.utils import *
from os.path import join, exists
from re import compile, finditer
from pathlm.knowledge_graphs.kg_macros import ENTITY_LIST

TAGS = {
    '<pi>': 'pi',
    '<rp>': 'rp',
    '<shared entity>': 'se',
    '<type of entity>': 'te',
    '<relation>': 're'
}

entities_ids: dict = {}

def fill_template(template_row_and_tags: list[tuple], path: str) -> str:
    template_row: list = list(template_row_and_tags[0])
    tags: tuple[str] = template_row_and_tags[1]
    for tag in tags:
        template_row[tag[1]] = f'<start_{TAGS[tag[0]]}> {path[tag[0]]} <end_{TAGS[tag[0]]}>'
        template_row[tag[1]+1:tag[2]] = [''] * (tag[2] - tag[1] - 1)

    return ''.join(template_row)

def get_type_of_entity(entity: str) -> str:
    if entity[0] == 'U':
        return 'USER'
    else:
        for entity_name, entity_ids in entities_ids.items():
            entity_num: str = entity[1:]
            if entity_num in entity_ids:
                return entity_name.upper()
        else:
            return 'ENTITY'

def explode_relation(relation_text: str) -> dict:
    return {
        '<pi>': relation_text[2],
        '<rp>': relation_text[-1],
        '<shared entity>': relation_text[-3],
        '<type of entity>': get_type_of_entity(relation_text[-3]),
        '<relation>': relation_text[-2]
    }

def preprocess_template(template_rows: list) -> list:
    pattern = compile(r'(<pi>|<rp>|<shared entity>|<type of entity>|<relation>){1,}')
    template_preprocessed_rows: list = []
    for i, row in enumerate(template_rows):
        row_tags: list = []
        print(f'Processing row {i+1} of {len(template_rows)}')
        for match in finditer(pattern, row):
            row_tags.append((match.group(), match.start(), match.end()))
        template_preprocessed_rows.append((row, row_tags))

    return template_preprocessed_rows

if __name__ == '__main__':
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--blank-template-file-name', '-iT', type=str, default='template0.txt',
                        help='Path to the blank template file')
    parser.add_argument('--raw-paths-file-name', '-iP', type=str, default='paths_end-to-end_250_3.txt',
                        help='Path to the raw paths file')
    parser.add_argument('--filled-template-file-name', '-oT', type=str, default='filled_template.txt',
                        help='Path to the filled template file')
    parser.add_argument('--dataset-name', '-d', type=str, default='ml1m',
                        help='Name of the dataset')
    parser.add_argument('--max-path', '-L', type=int, default=100,
                        help='Maximum number of path used to fill the template')
    args = parser.parse_args()

    template_dir_path: str = get_data_template_dir(args.dataset_name)
    blank_template_file_path: str = join(template_dir_path, args.blank_template_file_name)

    if not exists(blank_template_file_path):
        print(f'Error: {blank_template_file_path} does not exist')
        exit(1)

    print('Reading entities ids...')

    current_entities_for_dataset: list = ENTITY_LIST[args.dataset_name]
    for current_entity in current_entities_for_dataset:
        entity_ids_file_path: str = get_entity_ids_file_path(args.dataset_name, current_entity)
        ids_of_entities: list = read_entity_ids_file(entity_ids_file_path)
        entities_ids.update({current_entity: ids_of_entities})

    with open(blank_template_file_path) as template_file:
        blank_template_rows: list = template_file.readlines()

    preprocessed_template_rows: list = preprocess_template(blank_template_rows)
    raw_paths_dir_path: str = get_raw_paths_dir(args.dataset_name)
    raw_paths_file_path: str = join(raw_paths_dir_path, args.raw_paths_file_name)

    if not exists(raw_paths_file_path):
        print(f'Error: {raw_paths_file_path} does not exist')
        exit(1)

    with open(raw_paths_file_path) as raw_paths_file:
        raw_paths_rows: list = raw_paths_file.readlines()

    processed_paths_rows: list[tuple] = [
        (explode_relation(exploded_relation := row.strip().split(' ')), ' '.join(exploded_relation[:2]+[exploded_relation[-1]]))
        for row in raw_paths_rows[:args.max_path]
    ]

    filled_template_rows: list = [
        (fill_template(choice(preprocessed_template_rows), row[0]), row[1])
        for row in processed_paths_rows
    ]
    
    filled_template_file_path: str = join(get_filled_templates_dir(args.dataset_name), args.filled_template_file_name)
    with open(filled_template_file_path, 'w') as filled_template_file_obj:
        filled_template_file_obj.write('\n'.join([
            f'<start_rec> {line[1].strip()} <end_rec> <start_exp> {line[0].strip()[1:-1]} <end_exp>'
            for line in filled_template_rows
        ]))

    print(f'Saved successfully on {filled_template_file_path}!')
