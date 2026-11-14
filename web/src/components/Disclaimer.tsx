/**
 * The notice that appears on every page.
 *
 * Non-negotiable and rendered from the layout, not from a page, so a new page
 * cannot be added without it. Beacon is evaluated on synthetic benchmark
 * patients and has no clinical validity. Styled as a calm warning callout
 * rather than an alarm: it is always true, not an alert about the current run.
 */
export function Disclaimer() {
  return (
    <div className="mx-auto max-w-7xl px-4 pt-5 sm:px-6">
      <div
        role="note"
        className="border-t-4 border-nhs-warm bg-[#FFF8E1] px-4 py-3 text-[14px] leading-6 text-nhs-ink shadow-card"
      >
        <strong className="font-bold">Decision support, not medical advice.</strong>{" "}
        Beacon ranks clinical trials for review by a qualified person. It does not
        determine eligibility and must not be used to make care decisions. All
        patient data shown here is{" "}
        <strong className="font-bold">synthetic</strong> — generated for
        benchmarking, not taken from any real person.
      </div>
    </div>
  );
}
