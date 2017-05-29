
def create_raw_contrast_data(imgs, mask, raw_dir,
                             memory=Memory(cachedir=None),
                             n_jobs=1, batch_size=100):
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    # Selection of contrasts
    masker = MultiNiftiMasker(smoothing_fwhm=0,
                              mask_img=mask,
                              memory=memory,
                              memory_level=1,
                              n_jobs=n_jobs).fit()
    mask_img_file = os.path.join(raw_dir, 'mask_img.nii.gz')
    masker.mask_img_.to_filename(mask_img_file)

    batches = gen_batches(len(imgs), batch_size)

    data = np.empty((len(imgs), masker.mask_img_.get_data().sum()),
                    dtype=np.float32)
    for i, batch in enumerate(batches):
        print('Batch %i' % i)
        data[batch] = masker.transform(imgs['z_map'].values[batch])
    imgs = pd.DataFrame(data=data, index=imgs.index, dtype=np.float32)
    dump(imgs, join(raw_dir, 'imgs.pkl'))


def get_raw_contrast_data(raw_dir):
    mask_img = os.path.join(raw_dir, 'mask_img.nii.gz')
    masker = MultiRawMasker(smoothing_fwhm=0, mask_img=mask_img)
    masker.fit()
    imgs = load(join(raw_dir, 'imgs.pkl'))
    return masker, imgs


def build_design(datasets, datasets_dir, n_subjects):
    X = []
    for dataset in datasets:
        masker, this_X = get_raw_contrast_data(datasets_dir[dataset])
        subjects = this_X.index.get_level_values(
            'subject').unique().values.tolist()

        subjects = subjects[:n_subjects]
        X.append(this_X.loc[subjects])
    X = pd.concat(X, keys=datasets, names=['dataset'])

    return X, masker