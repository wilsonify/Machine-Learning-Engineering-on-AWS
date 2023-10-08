def load_model():
    sym_json = json_load(open('mx-mod-symbol.json'))
    sym_json_string = json_dumps(sym_json)

    model = gluon.nn.SymbolBlock(
        outputs=mxnet.sym.load_json(sym_json_string),
        inputs=mxnet.sym.var('data'))

    model.load_parameters(
        'mx-mod-0000.params',
        allow_missing=True
    )
    model.initialize()

    return model
