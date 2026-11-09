/**
 * The notice that appears on every page.
 *
 * Non-negotiable and rendered from the layout, not from a page, so a new page
 * cannot be added without it. Beacon is evaluated on synthetic benchmark
 * patients and has no clinical validity.
 */
export function Disclaimer() {
  return (
    <div
      role="note"
      className="border-b border-amber-300 bg-amber-100 px-4 py-2 text-center text-sm text-amber-950"
    >
      <strong className="font-semibold">
        Decision support, not medical advice.
      </strong>{" "}
      Beacon ranks clinical trials for review by a qualified person. It does not
      determine eligibility and must not be used to make care decisions. All
      patient data shown here is{" "}
      <strong className="font-semibold">synthetic</strong> — generated for
      benchmarking, not taken from any real person.
    </div>
  );
}
