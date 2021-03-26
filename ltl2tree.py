import nltk
import numpy as np
import random
import torch

from collections import defaultdict
from nltk import CFG
from nltk.parse import BottomUpLeftCornerChartParser


LTL_OPS = ['&', '|', 'G', 'F', 'X', 'U', '!']

OP2NARG = defaultdict(int)
OP2NARG.update({
        '&': 2, '|': 2, '!': 1,
        'G': 1, 'F': 1, 'X': 1, 'U': 2
    })

OP_1 = ['!', 'G', 'F', 'X']
OP_2 = ['&', '|', 'U']

LTL_GRAMMAR = """
    S -> LPAREN S RPAREN | TOP_1 S | S TOP_2 S | S OP_2 S | OP_1 S | TERM
    TOP_1 -> 'G' | 'F' | 'X'
    TOP_2 -> 'U'
    OP_1 -> '!'
    OP_2 -> '&' | '|'
    LPAREN -> '('
    RPAREN -> ')'
    TERM -> %s
"""

CRAFT_LTL_GRAMMAR = """
    S -> LPAREN EXPR RPAREN
    EXPR -> TOP_1 S | S TOP_2 S | S OP_2 S | OP_1 S | TERM
    TOP_1 -> 'G' | 'F'
    TOP_2 -> 'U'
    OP_1 -> '!'
    OP_2 -> '&' | '|'
    LPAREN -> '('
    RPAREN -> ')'
    TERM -> %s
"""



def get_ltl_grammar(symbols, env_name=''):
    if env_name == 'Craft':
        alphabet_str = ' | '.join(["'"+a+"'" for a in symbols if 'C_' not in a])
        grammar_str = CRAFT_LTL_GRAMMAR % alphabet_str
    else:
        alphabet_str = ' | '.join(["'"+a+"'" for a in symbols])
        grammar_str = LTL_GRAMMAR % alphabet_str
    grammar = CFG.fromstring(grammar_str)
    return grammar


def parse_ltl(formula, symbols, env_name=''):
    tokens = formula.replace('(', '( ').replace(')', ' )').replace('!', '! ').split()
    grammar = get_ltl_grammar(symbols, env_name)
    parser = BottomUpLeftCornerChartParser(grammar)
    trees = [tree for tree in parser.parse(tokens)]
    if len(trees) > 0:
        return trees[0]
    return None


class LTLTree(object):
    def __init__(self, val, id=0):
        self.value = val
        self.children = []
        self.parent = None
        self.size = 1
        self.depth = 0
        self.id = id

    def __repr__(self):
        return self.value


def get_node_val(tree):
    if len(tree) == 1 and type(tree[0]) != nltk.Tree:
        return tree[0]
    return None


def convert_ltl_tree(cfg_tree, id=0):
    i = 0; prev_branch = None; ltl_tree = None
    while i < len(cfg_tree):
        node_val = get_node_val(cfg_tree[i])
        if node_val in ['(', ')']:
            pass
        elif node_val in LTL_OPS:  # operators
            ltl_tree = LTLTree(node_val, id)
            id += 1
            if OP2NARG[node_val] == 1:
                child_ltl_tree, id = convert_ltl_tree(cfg_tree[i+1], id=id)
            elif OP2NARG[node_val] == 2:
                ltl_tree.children.append(prev_branch)
                prev_branch.parent = ltl_tree
                child_ltl_tree, id = convert_ltl_tree(cfg_tree[i+1], id=id)
            ltl_tree.children.append(child_ltl_tree)
            ltl_tree.size += sum([child.size for child in ltl_tree.children])
            ltl_tree.depth = max([child.depth for child in ltl_tree.children]) + 1
            child_ltl_tree.parent = ltl_tree
            i += 1
        elif node_val is None:
            ltl_tree, id = convert_ltl_tree(cfg_tree[i], id=id)
            prev_branch = ltl_tree
        else:  # symbols
            ltl_tree = LTLTree(node_val, id)
            id += 1
            prev_branch = ltl_tree
        i += 1
    return ltl_tree, id


def ltl_tree_str(ltl_tree):
    args = ', '.join([ltl_tree_str(ltl_tree.children[i]) for i in range(len(ltl_tree.children))])
    if ltl_tree.value in LTL_OPS:
        out = ltl_tree.value + '(' + args + ')'
    else:
        out = ltl_tree.value
    return out


def ltl_tree_with_annotation_str(ltl_tree, idx=0, ancestors=None, anno_maps=None):
    if ancestors is None:
        ancestors = []
    if anno_maps is None:
        anno_maps = {}
    ancestors.append(ltl_tree.value.split('_')[0])
    if ltl_tree.value in OP_2:
        left_out, idx, anno_maps = ltl_tree_with_annotation_str(ltl_tree.children[0], idx, ancestors, anno_maps)
        right_out, idx, anno_maps = ltl_tree_with_annotation_str(ltl_tree.children[1], idx, ancestors, anno_maps)
        out = '( ( {} {} {} ) & a_{} )'.format(left_out, ltl_tree.value, right_out, idx)
    elif ltl_tree.value in OP_1:
        in_out, idx, anno_maps = ltl_tree_with_annotation_str(ltl_tree.children[0], idx, ancestors, anno_maps)
        out = '( {} {} & a_{} )'.format(ltl_tree.value, in_out, idx)
    else:
        out = '( {} & a_{} )'.format(ltl_tree.value, idx)
    anno_maps['a_{}'.format(idx)] = set(ancestors)
    del ancestors[-1]
    return out, idx+1, anno_maps


def ltl2tree(formula, symbols, baseline=False):
    if baseline:
        return LTLTree('all')
    cfg_tree = parse_ltl(formula, symbols)
    if cfg_tree is not None:
        return convert_ltl_tree(cfg_tree)[0]
    return None


def ltlstr2template(formula):
    tokens = formula.split(' ')
    for i, token in enumerate(tokens):
        if token not in LTL_OPS and token not in ['(', ')']:
            tokens[i] = '_'
    return ' '.join(tokens)


def generate_ltl(cfg, cfactor=0.5, env_name=''):
    pcount = defaultdict(int)

    def weighted_choice(weights):
        rnd = random.random() * sum(weights)
        for i, w in enumerate(weights):
            rnd -= w
            if rnd < 0:
                return i

    def generate_sample(grammar, prod, frags):
        if prod in grammar._lhs_index:  # derivation
            derivations = grammar._lhs_index[prod]
            weights = []
            for prod in derivations:
                if prod in pcount:
                   weights.append(cfactor ** (pcount[prod]))
                else:
                    weights.append(1.0)
            # tend to not sample the already expanded productions
            derivation = derivations[weighted_choice(weights)]
            pcount[derivation] += 1
            for d in derivation._rhs:
                generate_sample(grammar, d, frags)
            pcount[derivation] -= 1
        elif prod in grammar._rhs_index:  # terminal
            prod = str(prod)
            frags.append(prod)

    frags = []
    generate_sample(cfg, cfg.start(), frags)
    return ' '.join(frags)


def ltl2onehot(formula, symbols):
    n_symbols = len(symbols)
    n_ops = len(LTL_OPS)
    # seq: alphabets -> ops -> parentheses
    out = torch.zeros(len(formula.split()),
                      n_symbols+n_ops+2)
    for i, token in enumerate(formula.split()):
        if token in symbols:
            out[i][symbols.index(token)] = 1
        elif token in LTL_OPS:
            out[i][n_symbols+LTL_OPS.index(token)] = 1
        elif token == '(':
            out[i][-2] = 1
        elif token == ')':
            out[i][-1] = 1
    return out


def replace_symbols(ltl, env_name=''):
    if env_name != 'Craft':  # do nothing if not craft
        return ltl
    tokens = ltl.split(' ')
    out = []
    for token in tokens:
        if token not in LTL_OPS and token not in ['(', ')']:
            out.append('C_' + token + ' U ' + token)
        else:
            out.append(token)
    return ' '.join(out)


if __name__ == '__main__':
    ltl_formula = '(((G a) & (F b)) U ((X b) & (b U a)))'
    # parse ltl formula to cfg tree and then convert to an expression tree
    cfg_tree = parse_ltl(ltl_formula, ['a', 'b'])
    print(cfg_tree)
    ltl_tree, idx = convert_ltl_tree(cfg_tree)
    print(ltl_tree_str(ltl_tree))
    # use a function to combine the above two steps, should give the same tree
    ltl_tree = ltl2tree(ltl_formula, ['a', 'b'])
    print(ltl_tree_str(ltl_tree))
    # generate a new formula using the predefined grammar
    cfg = get_ltl_grammar(['a', 'b'])
    print(generate_ltl(cfg))
