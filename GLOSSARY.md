# Glossary

**task_type**: What the user expects to receive back from a prompt. One of six values. Defined in `docs/labeling-guidelines.md`.

**complexity**: The effort needed to do the task well, not the prompt's length. `low`, `medium` or `high`.

**cell**: One `task_type` × `complexity` combination. There are 18.

**target cell**: The cell a synthetic prompt was asked to be written for. It is the prompt's construction label.

**relabel**: An independent label for a prompt from a model other than the generator, assigned without seeing the target cell.

**disagreement**: A prompt whose target cell and relabel differ on `task_type` or `complexity`.

**borderline**: A prompt where labels disagree. The pool for the human review.
