# Mathematics of Low-Rank Geometric Updates

## 1. The Trace ($\Delta$)
For a given layer $l$, input $x$, and output $y$:
$$ \Delta = y - x $$
This represents the vector direction of the model's computation.

## 2. Low-Rank Metric Tensor ($G$)
A full Metric Tensor for a dimension $d=4096$ is too large ($16M$ entries). We approximate it using the top-$k$ principal components ($k=32$) of the traces found near a specific Concept Node.

$$ G \approx I + U \Lambda U^T $$
*   $U$: A $4096 \times 32$ matrix (Eigenvectors/Principal Directions).
*   $\Lambda$: A $32 \times 32$ diagonal matrix (Eigenvalues/Strengths).

## 3. The Natural Gradient (The "Smart" Delta)
Standard SGD updates weights in the direction of steepest descent in *Euclidean* space. We want steepest descent in *Riemannian* space (taking knowledge density into account). This requires $G^{-1}$.

Using the **Sherman-Morrison-Woodbury Identity**, we invert the low-rank metric efficiently:

$$ G^{-1} \approx I - U ( \Lambda^{-1} + U^T U )^{-1} U^T $$

*   The term in the middle is a $32 \times 32$ matrix. Inversion is trivial.
*   This calculation transforms a raw trace $\Delta$ into a **Natural Delta** $\tilde{\Delta}$.

## 4. Pole Ladder Transport
When a REM cycle generates a trace $\Delta_{dream}$ at a location $P_{dream}$ that is distant from the concept's centroid $P_{anchor}$, we must transport the vector before learning from it.

$$ \Delta_{transported} = \text{Reflect}(\Delta_{dream}, \text{Midpoint}(P_{dream}, P_{anchor})) $$

This approximates Parallel Transport along the geodesic without calculating Christoffel symbols.

## 5. The Hebbian Update Rule
We update the LoRA adapter weights $W_{lora}$ using the Natural Delta and the Input.

$$ \Delta W_{lora} = \eta \cdot \sum_{i=1}^{N} (\tilde{\Delta}_i \otimes x_i) $$

*   $\eta$: Learning Rate.
*   $\tilde{\Delta}_i$: The Natural Gradient (curved-corrected trace).
*   $x_i$: The input activation that caused the trace.
*   $\otimes$: Outer product.

This aligns the Student's internal weights with the "Map" created by the Teacher, respecting the existing topology of the Student's mind.
