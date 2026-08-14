import argparse
import torch
from tqdm import tqdm
from stablediff.dataset import from_path_split
from stablediff.params import params_simple
from stablediff.CVAE import CVAE

def train_epoch(vae, device, dataloader, optimizer):
    """Run one training epoch and return per-sample loss, reconstruction error, KL."""

    vae.train()

    train_loss = 0.0
    reconstruction_error_epoch = 0.0
    kl_error_epoch = 0.0

    for batch in tqdm(dataloader):

        x = batch["data"].to(device)

        optimizer.zero_grad()

        x_hat = vae(x)

        reconstruction_error = (
            torch.abs(x - x_hat) ** 2
        ).mean()

        # KL computed by encoder during forward pass
        kl_error = vae.encoder.kl.mean()

        loss = reconstruction_error + kl_error

        loss.backward()
        optimizer.step()

        reconstruction_error_epoch += reconstruction_error.item()
        kl_error_epoch += kl_error.item()
        train_loss += loss.item()

    len_dataset = len(dataloader.dataset)

    return (
        train_loss / len_dataset,
        reconstruction_error_epoch / len_dataset,
        kl_error_epoch / len_dataset
    )

def test_epoch(vae, device, dataloader):
    """Run one validation epoch and return per-sample loss."""

    vae.eval()

    val_loss = 0.0

    with torch.no_grad():

        for batch in dataloader:

            x = batch["data"].to(device)

            x_hat = vae(x)

            reconstruction_error = (
                torch.abs(x - x_hat) ** 2
            ).mean()

            kl_error = vae.encoder.kl.mean()

            loss = reconstruction_error + kl_error

            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)

    vae = CVAE(latent_dims=args.latent_dims).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader, valid_loader = from_path_split(params_simple)

    history = {"train_loss": [], "reconstruction_error": [], "kl_error": [], "val_loss": []}
    for epoch in range(args.epochs):
        train_loss, reconstruction_error, kl_error = train_epoch(vae, device, train_loader, optimizer)
        val_loss = test_epoch(vae, device, valid_loader)

        history["train_loss"].append(train_loss)
        history["reconstruction_error"].append(reconstruction_error)
        history["kl_error"].append(kl_error)
        history["val_loss"].append(val_loss)

        print('\n EPOCH {}/{} \t train loss {:.3f} (recon error {:.3f}, kl error {:.3f}) \t val loss {:.3f}'.format(
            epoch + 1, args.epochs, train_loss, reconstruction_error, kl_error, val_loss))

    #save model
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": vae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "latent_dims": args.latent_dims,
        },
        args.save_dir
    )

    print(f"Model saved to: {args.save_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--latent_dims", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--save_dir", type=str, default=params_simple.cvae_model_dir)

    
    args = parser.parse_args()
    main(args)