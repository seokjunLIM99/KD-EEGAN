
def create_model_teacher(opt):
    model = None
    print(opt.model)
    if opt.model == 'single':
        # assert(opt.dataset_mode == 'unaligned')
        from .single_model_teacher import SingleModel_t
        model = SingleModel_t()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
