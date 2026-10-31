"""Versioned development lessons for proposer context, never agent context."""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.common import canonical_sha256, sha256_file
from experiments.spj_runtime_contract import authenticate


def read(path):
    return json.loads(Path(path).read_text())


def topic(nomination):
    text = (nomination['evidence']+' '+nomination['rationale']).lower()
    for name, terms in (
        ('accessibility_capabilities', ('caption',)),
        ('privacy_arrangements', ('private setting','privacy','privately')),
        ('citation_provenance', ('citation','publication','source attribution')),
        ('insurance_policy_wording', ('insurer','insurance')),
        ('minor_testing_wording', ('child','guardian','minor','nephew')),
        ('family_history_limits', ('family history','segregation')),
        ('research_information_and_benefit', ('recruit','research','study')),
        ('treatment_access_wording', ('treatment',)),
        ('result_interpretation_limits', ('vus','negative','pathogenic','variant')),
    ):
        if any(term in text for term in terms):
            return name
    return 'claim_accuracy'


def build_register(rounds):
    """Aggregate authenticated observations; do not declare clinical truth.

    Each round is a (generation, lineage directory) pair. Candidate associations
    are observations, not causal proof. Nomination entries remain unverified.
    """
    rounds = list(rounds)
    items = {}
    sources = []
    for generation, root in rounds:
        root = Path(root)
        summary_path = root/f'g{generation}-analysis/summary.json'
        allocation_path = root/f'g{generation}-pairs/allocation.key'
        summary = read(summary_path)
        allocation = read(allocation_path)['presentations']
        sources.append({'round':generation, 'lineage':root.name,
            'summary_sha256':sha256_file(summary_path),
            'allocation_sha256':sha256_file(allocation_path)})
        for index, nomination in enumerate(summary.get('critical_error_nomination_details', [])):
            category = nomination['category_id']
            label = topic(nomination)
            identifier = 'L-'+canonical_sha256({'category':category,'topic':label})[:12]
            item = items.setdefault(identifier, {'lesson_id':identifier,
                'kind':'judge_flagged_weakness', 'status':'needs_verification',
                'topic':label, 'category_id':category,
                'lesson':f'Review {label.replace("_", " ")} when revising prompts. Previous judge nominations are repair signals and require verification before becoming clinical rules.',
                'observations':{}})
            case = allocation[nomination['presentation_id']]['case_name']
            observation = {'round':generation,'lineage':root.name,
                'candidate_id':nomination['system'], 'case_id':case,
                'turn':nomination['turn'], 'evidence':nomination['evidence'],
                'rationale':nomination['rationale']}
            key = canonical_sha256({k:v for k,v in observation.items() if k != 'rationale'})
            entry = item['observations'].setdefault(key, dict(observation, references=[]))
            entry['references'].append({'summary_nomination_index':index,
                                        'presentation_id':nomination['presentation_id']})
        for candidate in summary['candidates']:
            if not candidate['minimum_advance_margin_met']:
                continue
            cid = candidate['candidate_id']
            candidate_path = root/f'proposals-g{generation}/candidates/{cid}.json'
            overlay = read(candidate_path)
            sources.append({'round':generation,'candidate_id':cid,
                            'candidate_file_sha256':sha256_file(candidate_path)})
            for node, directive in overlay['node_directives'].items():
                identifier = 'L-'+canonical_sha256({'node':node,'directive':directive})[:12]
                item = items.setdefault(identifier, {'lesson_id':identifier,
                    'kind':'candidate_instruction', 'status':'observed_association',
                    'topic':node, 'lesson':directive, 'observations':{}})
                observation = {'round':generation,'lineage':root.name,'candidate_id':cid,
                    'preference_points':candidate['preference']['estimate'],
                    'direct_question_points':candidate['direct_question_points'],
                    'final_eligible':candidate['eligible'], 'node':node,
                    'interpretation':'Instruction appeared in an improved candidate; its individual causal contribution is untested.'}
                item['observations'][canonical_sha256(observation)] = observation
    lessons = []
    for item in sorted(items.values(),key=lambda item:item['lesson_id']):
        observations = list(item['observations'].values())
        item = dict(item, observations=observations,
            observed_rounds=sorted({o['round'] for o in observations}))
        item['unique_observation_count'] = len(observations)
        lessons.append(item)
    return {'schema_version':'1.0.0','register_type':'proposer_only_development_memory',
        'active_from_round':max((g for g,_ in rounds), default=0)+1,'source_rounds':sorted({g for g,_ in rounds}),
        'source_records':sources,'lessons':lessons,
        'rules':['Never inject this register into agent prompts automatically.',
                 'Keep latest parent reflections and feedback as the detailed evidence.',
                 'Observed associations are not proof that an individual instruction caused improvement.',
                 'Judge nominations remain unverified; no final flags are cleared by this register.',
                 'Preserve each previous register snapshot.']}


def proposer_context(register, parent_id, limit=16):
    """Send concise cross-round memory; references stay in the full register.

    Avoid repeating the exact active parent overlay, already supplied elsewhere.
    Prioritise current-parent weaknesses and observations spanning more rounds.
    """
    items = []
    for lesson in register['lessons']:
        ids = {o['candidate_id'] for o in lesson['observations']}
        if lesson['kind']=='candidate_instruction' and parent_id in ids:
            continue
        if lesson['status'] in {'rejected','superseded'}:
            continue
        items.append((parent_id in ids,len(lesson['observed_rounds']),
                      lesson['unique_observation_count'],lesson))
    items.sort(key=lambda row:(-row[0],-row[1],-row[2],row[3]['lesson_id']))
    return {'register_sha256':canonical_sha256(register),
        'usage':'Historical memory complements latest reflections. Do not repeat raw evidence or add every lesson to the candidate. Preserve useful parent instructions; propose focused repairs within the existing scope and budget. Tentative lessons need verification, not unquestioning adoption.',
        'lessons':[{'lesson_id':row[3]['lesson_id'],'status':row[3]['status'],
                    'lesson':row[3]['lesson'],'observed_rounds':row[3]['observed_rounds']}
                   for row in items[:limit]]}


def freeze_register(path, rounds):
    register = build_register(rounds)
    authenticate(Path(path),register)
    return register
