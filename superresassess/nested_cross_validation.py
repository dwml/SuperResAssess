from superresassess.assessment_base import AssessmentMethod


class NestedCrossValidation(AssessmentMethod):
    def _setup_file_splitting(
        self, train_val_test_ratio: tuple[float, float, float], n_internal_images: int
    ): ...

    def assess(self) -> None:
        ...
        # for ii in range(len(self._outer_folds)):
        #    for jj in range(len(self._inner_folds)):
        #        continue
